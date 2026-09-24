"""Generate a labeled eval set by templating real KG triples into questions (1-hop and 2-hop),
plus a set of out-of-scope questions about entities that do not exist in the KG (to measure
hallucination rate: does the system fabricate an answer instead of refusing?).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(13)

ONE_HOP_TEMPLATES = {
    ("package", "module", "CONTAINS"): ("Which package contains the module `{dst}`?", "src"),
    ("module", "class", "CONTAINS"): ("Which module defines the class `{dst}`?", "src"),
    ("module", "function", "CONTAINS"): ("Which module defines the function `{dst}`?", "src"),
    ("class", "function", "CONTAINS"): ("Which class defines the method `{dst}`?", "src"),
    ("module", "module", "IMPORTS"): ("What module does `{src}` import?", "dst"),
    ("package", "module", "IMPORTS"): ("What does the module `{src}` import?", "dst"),
    ("class", "class", "INHERITS"): ("What class does `{src}` inherit from?", "dst"),
    ("function", "function", "CALLS"): ("What function does `{src}` call?", "dst"),
}


def build_eval_set(cache_dir: Path, n_per_template: int = 6, n_two_hop: int = 14, n_out_of_scope: int = 15) -> list[dict]:
    nodes = {n["id"]: n for n in json.loads((cache_dir / "nodes.json").read_text())}
    edges = json.loads((cache_dir / "edges.json").read_text())

    by_type_key: dict[tuple, list[dict]] = {}
    for e in edges:
        src_t, dst_t = nodes[e["src"]]["type"], nodes[e["dst"]]["type"]
        key = (src_t, dst_t, e["type"])
        by_type_key.setdefault(key, []).append(e)

    examples = []

    # --- 1-hop templated questions, sampled across every (src_type, dst_type, relation) combo ---
    for key, (template, answer_field) in ONE_HOP_TEMPLATES.items():
        pool = by_type_key.get(key, [])
        if not pool:
            continue
        sample = random.sample(pool, min(n_per_template, len(pool)))
        for e in sample:
            answer = e[answer_field]
            question = template.format(src=e["src"], dst=e["dst"])
            examples.append({
                "question": question,
                "gold_answer": answer,
                "gold_triples": [[e["src"], e["type"], e["dst"]]],
                "hops": 1,
                "category": "answerable",
            })

    # --- 2-hop chains: A --CONTAINS--> B --CONTAINS--> C  (package -> module -> class/function) ---
    contains_edges = [e for e in edges if e["type"] == "CONTAINS"]
    by_src = {}
    for e in contains_edges:
        by_src.setdefault(e["src"], []).append(e)

    two_hop_contains = []
    for e1 in contains_edges:
        if nodes[e1["src"]]["type"] != "package":
            continue
        for e2 in by_src.get(e1["dst"], []):
            if nodes[e2["dst"]]["type"] in ("class", "function"):
                two_hop_contains.append((e1, e2))

    for e1, e2 in random.sample(two_hop_contains, min(n_two_hop // 2, len(two_hop_contains))):
        kind = "class" if nodes[e2["dst"]]["type"] == "class" else "function"
        question = f"Which package ultimately contains the {kind} `{e2['dst']}` (via its module)?"
        examples.append({
            "question": question,
            "gold_answer": e1["src"],
            "gold_triples": [[e1["src"], e1["type"], e1["dst"]], [e2["src"], e2["type"], e2["dst"]]],
            "hops": 2,
            "category": "answerable",
        })

    # --- 2-hop chains: A --IMPORTS--> B --CONTAINS--> C  (module imports module that defines class) ---
    import_edges = [e for e in edges if e["type"] == "IMPORTS"]
    two_hop_import = []
    for e1 in import_edges:
        for e2 in by_src.get(e1["dst"], []):
            if nodes[e2["dst"]]["type"] == "class":
                two_hop_import.append((e1, e2))

    for e1, e2 in random.sample(two_hop_import, min(n_two_hop - n_two_hop // 2, len(two_hop_import))):
        question = f"Name a class defined in a module that `{e1['src']}` imports."
        examples.append({
            "question": question,
            "gold_answer": e2["dst"],
            "gold_triples": [[e1["src"], e1["type"], e1["dst"]], [e2["src"], e2["type"], e2["dst"]]],
            "hops": 2,
            "category": "answerable",
        })

    # --- out-of-scope: fabricated entities that do not exist in the KG ---
    fabricated = [
        "flask.nonexistent_module.FakeClass", "werkzeug.quantum_routing.TeleportHandler",
        "jinja2.blockchain.SmartTemplate", "click.astro.StarCommand", "itsdangerous.timetravel.Paradox",
        "markupsafe.holograms.Renderer", "numpy.core.ndarray", "pandas.DataFrame.pivot_table",
        "requests.sessions.Session", "flask.app.Flask.teleport", "werkzeug.wrappers.Response.levitate",
        "django.db.models.Model", "click.core.Command.self_destruct", "jinja2.environment.Environment.time_travel",
        "flask.cli.ScriptInfo.predict_future",
    ]
    for name in fabricated[:n_out_of_scope]:
        entity_kind = "class" if name[-1].isupper() or name.split(".")[-1][0].isupper() else "function"
        examples.append({
            "question": f"What does `{name}` inherit from?" if entity_kind == "class"
            else f"What does the function `{name}` call?",
            "gold_answer": None,
            "gold_triples": [],
            "hops": 0,
            "category": "out_of_scope",
        })

    random.shuffle(examples)
    return examples


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = repo_root / "data" / "cache"
    eval_set = build_eval_set(cache_dir)
    out_path = repo_root / "data" / "eval_set.json"
    out_path.write_text(json.dumps(eval_set, indent=2))
    n_answerable = sum(1 for e in eval_set if e["category"] == "answerable")
    n_oos = sum(1 for e in eval_set if e["category"] == "out_of_scope")
    print(f"wrote {len(eval_set)} questions ({n_answerable} answerable, {n_oos} out-of-scope) -> {out_path}")
