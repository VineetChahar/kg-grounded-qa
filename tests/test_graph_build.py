"""Unit tests for graph construction against tiny synthetic packages (not the real cloned repos)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kg_grounded_qa.graph_build import PACKAGES, RepoParser, build_graph, graph_stats


def _write_pkg(tmp_path: Path, repo_name: str, package: str, files: dict[str, str]) -> Path:
    repo_dir = tmp_path / repo_name
    src_root = repo_dir / "src" / package
    src_root.mkdir(parents=True)
    (src_root / "__init__.py").write_text("")
    for rel_path, content in files.items():
        p = src_root / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return repo_dir


@pytest.fixture
def synthetic_repo_dir(tmp_path):
    return _write_pkg(tmp_path, "demo", "demo", {
        "core.py": '''
"""Core module."""

class Base:
    """Base class."""
    def greet(self):
        return "hi"

class Child(Base):
    """Child class."""
    def greet_twice(self):
        return self.greet() + self.greet()

def helper():
    """A helper function."""
    return Base().greet()
''',
        "utils.py": '''
from .core import Base

def use_helper():
    b = Base()
    return b.greet()
''',
    })


def test_parser_collects_modules_classes_functions(synthetic_repo_dir):
    parser = RepoParser(synthetic_repo_dir, "demo")
    nodes, edges = parser.parse()

    assert "demo" in nodes and nodes["demo"].type == "package"
    assert "demo.core" in nodes and nodes["demo.core"].type == "module"
    assert "demo.core.Base" in nodes and nodes["demo.core.Base"].type == "class"
    assert "demo.core.Base.greet" in nodes and nodes["demo.core.Base.greet"].type == "function"
    assert "demo.core.helper" in nodes and nodes["demo.core.helper"].type == "function"


def test_contains_edges(synthetic_repo_dir):
    parser = RepoParser(synthetic_repo_dir, "demo")
    _, edges = parser.parse()
    assert ("demo", "demo.core", "CONTAINS") in edges
    assert ("demo.core", "demo.core.Base", "CONTAINS") in edges
    assert ("demo.core.Base", "demo.core.Base.greet", "CONTAINS") in edges


def test_inherits_edge_resolved(synthetic_repo_dir):
    parser = RepoParser(synthetic_repo_dir, "demo")
    _, edges = parser.parse()
    assert ("demo.core.Child", "demo.core.Base", "INHERITS") in edges


def test_calls_edge_resolved_within_class(synthetic_repo_dir):
    parser = RepoParser(synthetic_repo_dir, "demo")
    _, edges = parser.parse()
    assert ("demo.core.Child.greet_twice", "demo.core.Base.greet", "CALLS") in edges


def test_imports_edge_relative(synthetic_repo_dir):
    parser = RepoParser(synthetic_repo_dir, "demo")
    _, edges = parser.parse()
    assert ("demo.utils", "demo.core", "IMPORTS") in edges


def test_docstring_captured(synthetic_repo_dir):
    parser = RepoParser(synthetic_repo_dir, "demo")
    nodes, _ = parser.parse()
    assert nodes["demo.core"].docstring == "Core module."
    assert nodes["demo.core.Base"].docstring == "Base class."


def test_ambiguous_call_not_resolved(tmp_path):
    """Two functions named `run` in different unrelated modules -> CALLS should stay unresolved."""
    repo_dir = _write_pkg(tmp_path, "demo2", "demo2", {
        "a.py": "def run():\n    return 1\n",
        "b.py": "def run():\n    return 2\n",
        "c.py": "def entry():\n    return run()\n",
    })
    parser = RepoParser(repo_dir, "demo2")
    _, edges = parser.parse()
    calls_from_entry = [e for e in edges if e[0] == "demo2.c.entry" and e[2] == "CALLS"]
    assert calls_from_entry == []


def test_build_graph_stats_shape(tmp_path):
    repos_dir = tmp_path / "repos"
    _write_pkg(repos_dir, "demo", "demo", {"core.py": "def f():\n    pass\n"})
    graph = build_graph(repos_dir, packages={"demo": "demo"})
    stats = graph_stats(graph)
    assert stats["num_nodes"] > 0
    assert "node_type_counts" in stats
    assert "edge_type_counts" in stats
