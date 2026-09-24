"""Build a package/module/class/function knowledge graph from cloned repos via static AST analysis.

Node types: package, module, class, function (covers both free functions and methods).
Edge types: CONTAINS, IMPORTS, INHERITS, CALLS.

CALLS and IMPORTS edges are only added when the target resolves unambiguously to a
node already in the graph (see resolve_call / resolve_import) - this trades recall
for precision, since a wrong edge silently corrupts both GNN training and retrieval.
"""

from __future__ import annotations

import ast
import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("graph_build")

# repo dir name -> importable package name
PACKAGES = {
    "click": "click",
    "flask": "flask",
    "itsdangerous": "itsdangerous",
    "jinja": "jinja2",
    "markupsafe": "markupsafe",
    "werkzeug": "werkzeug",
}

MAX_DOC_LEN = 200


@dataclass
class Node:
    id: str
    type: str  # package | module | class | function
    name: str
    package: str
    docstring: str = ""
    signature: str = ""
    filepath: str = ""
    lineno: int = 0


def _truncate(text: str | None, n: int = MAX_DOC_LEN) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text[:n]


def _signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = [a.arg for a in fn.args.args]
    return f"({', '.join(args)})"


class RepoParser:
    """Parses one cloned repo's src/<package> tree into nodes + edges."""

    def __init__(self, repo_dir: Path, package: str):
        self.repo_dir = repo_dir
        self.package = package
        self.src_root = repo_dir / "src"
        self.nodes: dict[str, Node] = {}
        self.edges: list[tuple[str, str, str]] = []  # (src, dst, type)
        # short_name -> list of qualified ids, for CALLS resolution
        self.symbol_table: dict[str, list[str]] = defaultdict(list)
        # module dotted name -> file path, for IMPORTS resolution
        self.module_ids: set[str] = set()

    def module_name_for(self, py_file: Path) -> str:
        rel = py_file.relative_to(self.src_root).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts:
            return self.package
        return ".".join(parts)

    def discover_modules(self) -> list[Path]:
        return sorted(self.src_root.glob("**/*.py"))

    def pass1_collect_defs(self, py_files: list[Path]) -> None:
        """First pass: register package/module/class/function nodes and build the symbol table."""
        pkg_node = Node(id=self.package, type="package", name=self.package, package=self.package)
        self.nodes[self.package] = pkg_node

        for py_file in py_files:
            mod_name = self.module_name_for(py_file)
            mod_id = mod_name
            self.module_ids.add(mod_id)
            rel_path = str(py_file.relative_to(self.repo_dir))
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            except (SyntaxError, UnicodeDecodeError) as e:
                logger.warning(f"skip unparsable {py_file}: {e}")
                continue

            is_root_init = mod_id == self.package
            if is_root_init:
                # root __init__.py IS the package node; don't create a duplicate module
                # node with the same id, and don't add a package->itself CONTAINS edge.
                if not self.nodes[self.package].docstring:
                    self.nodes[self.package].docstring = _truncate(ast.get_docstring(tree))
            else:
                self.nodes[mod_id] = Node(
                    id=mod_id, type="module", name=mod_name.split(".")[-1],
                    package=self.package, docstring=_truncate(ast.get_docstring(tree)),
                    filepath=rel_path,
                )
                self.edges.append((self.package, mod_id, "CONTAINS"))

            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    self._add_class(node, mod_id, mod_name, rel_path)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self._add_function(node, mod_id, mod_name, mod_id, rel_path, is_method=False)

    def _add_class(self, node: ast.ClassDef, mod_id: str, mod_name: str, rel_path: str) -> None:
        class_id = f"{mod_name}.{node.name}"
        self.nodes[class_id] = Node(
            id=class_id, type="class", name=node.name, package=self.package,
            docstring=_truncate(ast.get_docstring(node)), filepath=rel_path, lineno=node.lineno,
        )
        self.edges.append((mod_id, class_id, "CONTAINS"))
        self.symbol_table[node.name].append(class_id)

        for base in node.bases:
            base_name = self._name_of(base)
            if base_name:
                # deferred: resolved to a real id in pass2 once all classes are known
                self.edges.append((class_id, f"__INHERITS_NAME__:{base_name}", "INHERITS"))

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._add_function(item, class_id, mod_name, mod_id, rel_path, is_method=True)

    def _add_function(self, node, parent_id, mod_name, mod_id, rel_path, is_method: bool) -> None:
        fn_id = f"{parent_id}.{node.name}"
        self.nodes[fn_id] = Node(
            id=fn_id, type="function", name=node.name, package=self.package,
            docstring=_truncate(ast.get_docstring(node)), signature=_signature(node),
            filepath=rel_path, lineno=node.lineno,
        )
        self.edges.append((parent_id, fn_id, "CONTAINS"))
        self.symbol_table[node.name].append(fn_id)
        # stash the body + owning scope for CALLS resolution in pass 2
        self._pending_calls = getattr(self, "_pending_calls", [])
        self._pending_calls.append((fn_id, mod_id, parent_id if is_method else None, node))

    @staticmethod
    def _name_of(expr: ast.expr) -> str | None:
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.Attribute):
            return expr.attr
        return None

    def pass2_resolve_inherits(self) -> None:
        resolved = []
        for src, dst, typ in self.edges:
            if typ == "INHERITS" and dst.startswith("__INHERITS_NAME__:"):
                base_name = dst.split(":", 1)[1]
                candidates = self.symbol_table.get(base_name, [])
                class_candidates = [c for c in candidates if self.nodes.get(c, Node("", "", "", "")).type == "class"]
                if len(class_candidates) == 1:
                    resolved.append((src, class_candidates[0], "INHERITS"))
                # ambiguous or external (e.g. `object`, `Exception`) -> drop
            else:
                resolved.append((src, dst, typ))
        self.edges = resolved

    def pass3_resolve_calls(self) -> None:
        for fn_id, mod_id, class_id, fn_node in getattr(self, "_pending_calls", []):
            local_scope_name = self.nodes[class_id].name if class_id else None
            for call in ast.walk(fn_node):
                if not isinstance(call, ast.Call):
                    continue
                callee_name = self._name_of(call.func)
                if not callee_name or callee_name not in self.symbol_table:
                    continue
                target = self._resolve_symbol(callee_name, mod_id, class_id)
                if target and target != fn_id:
                    self.edges.append((fn_id, target, "CALLS"))

    def _resolve_symbol(self, name: str, mod_id: str, class_id: str | None) -> str | None:
        candidates = self.symbol_table.get(name, [])
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        # prefer a candidate defined on the same class, then same module, else drop (ambiguous)
        if class_id:
            same_class = [c for c in candidates if c.startswith(f"{class_id}.") and c.count(".") == class_id.count(".") + 1]
            if len(same_class) == 1:
                return same_class[0]
        same_module = [c for c in candidates if c.startswith(f"{mod_id}.")]
        if len(same_module) == 1:
            return same_module[0]
        return None

    def pass4_resolve_imports(self, py_files: list[Path]) -> None:
        for py_file in py_files:
            mod_name = self.module_name_for(py_file)
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self._maybe_add_import(mod_name, alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.level and node.level > 0:
                        target = self._resolve_relative(mod_name, node.level, node.module)
                    else:
                        target = node.module
                    if target:
                        self._maybe_add_import(mod_name, target)

    def _resolve_relative(self, mod_name: str, level: int, module: str | None) -> str | None:
        parts = mod_name.split(".")
        base = parts[: max(len(parts) - level, 0)]
        if module:
            base.append(module)
        return ".".join(base) if base else None

    def _maybe_add_import(self, src_mod: str, target: str) -> None:
        if src_mod == target:
            return
        if target in self.module_ids:
            self.edges.append((src_mod, target, "IMPORTS"))
            return
        # cross-package: check if target is (a prefix of) another known package's module
        for pkg in PACKAGES.values():
            if target == pkg or target.startswith(pkg + "."):
                # find the closest known module id for this target
                if target in self.module_ids:
                    self.edges.append((src_mod, target, "IMPORTS"))
                elif pkg in self.nodes:
                    self.edges.append((src_mod, pkg, "IMPORTS"))
                return

    def parse(self) -> tuple[dict[str, Node], list[tuple[str, str, str]]]:
        py_files = self.discover_modules()
        self.pass1_collect_defs(py_files)
        self.pass2_resolve_inherits()
        self.pass3_resolve_calls()
        self.pass4_resolve_imports(py_files)
        return self.nodes, self.edges


def build_graph(repos_dir: Path, packages: dict[str, str] | None = None) -> nx.MultiDiGraph:
    packages = packages if packages is not None else PACKAGES
    all_nodes: dict[str, Node] = {}
    all_edges: list[tuple[str, str, str]] = []
    # module_ids across ALL packages, needed for cross-repo import resolution
    global_module_ids: set[str] = set()

    parsers = []
    for repo_name, package in packages.items():
        repo_dir = repos_dir / repo_name
        if not repo_dir.exists():
            logger.warning(f"missing repo dir {repo_dir}, skipping")
            continue
        parser = RepoParser(repo_dir, package)
        parsers.append(parser)

    # pass1-3 per repo (defs + inherits + calls are repo-local)
    for parser in parsers:
        py_files = parser.discover_modules()
        parser.pass1_collect_defs(py_files)
        parser.pass2_resolve_inherits()
        parser.pass3_resolve_calls()
        global_module_ids.update(parser.module_ids)
        all_nodes.update(parser.nodes)
        all_edges.extend(parser.edges)

    # pass4 (imports) needs the GLOBAL module id set for cross-repo resolution
    for parser in parsers:
        parser.module_ids = global_module_ids
        checkpoint = len(parser.edges)
        py_files = parser.discover_modules()
        parser.pass4_resolve_imports(py_files)
        all_edges.extend(parser.edges[checkpoint:])

    # dedupe imports (pass4 above can double count since edges list already had earlier passes)
    seen = set()
    deduped_edges = []
    for e in all_edges:
        if e in seen:
            continue
        seen.add(e)
        deduped_edges.append(e)

    g = nx.MultiDiGraph()
    for nid, n in all_nodes.items():
        g.add_node(nid, **asdict(n))
    for src, dst, typ in deduped_edges:
        if src in all_nodes and dst in all_nodes:
            g.add_edge(src, dst, type=typ)

    return g


def graph_stats(g: nx.MultiDiGraph) -> dict:
    type_counts = Counter(nx.get_node_attributes(g, "type").values())
    edge_type_counts = Counter(d["type"] for _, _, d in g.edges(data=True))
    degrees = [d for _, d in g.degree()]
    return {
        "num_nodes": g.number_of_nodes(),
        "num_edges": g.number_of_edges(),
        "node_type_counts": dict(type_counts),
        "edge_type_counts": dict(edge_type_counts),
        "avg_degree": sum(degrees) / len(degrees) if degrees else 0,
        "max_degree": max(degrees) if degrees else 0,
        "min_degree": min(degrees) if degrees else 0,
    }


def save_graph(g: nx.MultiDiGraph, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(g, out_dir / "graph.graphml")

    nodes_out = [dict(d) for _, d in g.nodes(data=True)]
    edges_out = [dict(src=u, dst=v, **d) for u, v, d in g.edges(data=True)]
    (out_dir / "nodes.json").write_text(json.dumps(nodes_out, indent=2))
    (out_dir / "edges.json").write_text(json.dumps(edges_out, indent=2))

    stats = graph_stats(g)
    (out_dir / "graph_stats.json").write_text(json.dumps(stats, indent=2))
    logger.info(json.dumps(stats, indent=2))


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    repos_dir = repo_root / "data" / "repos"
    out_dir = repo_root / "data" / "cache"
    graph = build_graph(repos_dir)
    save_graph(graph, out_dir)
