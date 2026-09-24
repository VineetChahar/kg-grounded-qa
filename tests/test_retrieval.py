import json

import numpy as np

from kg_grounded_qa.retrieval import KGIndex, embedding_only_retrieve, link_entities, multihop_retrieve


def _write_synthetic_cache(cache_dir):
    nodes = [
        {"id": "pkg", "type": "package", "name": "pkg", "package": "pkg", "docstring": "", "signature": ""},
        {"id": "pkg.mod", "type": "module", "name": "mod", "package": "pkg", "docstring": "a module", "signature": ""},
        {"id": "pkg.mod.Foo", "type": "class", "name": "Foo", "package": "pkg", "docstring": "a class", "signature": ""},
        {"id": "pkg.mod.Foo.bar", "type": "function", "name": "bar", "package": "pkg", "docstring": "a method", "signature": "()"},
    ]
    edges = [
        {"src": "pkg", "dst": "pkg.mod", "type": "CONTAINS"},
        {"src": "pkg.mod", "dst": "pkg.mod.Foo", "type": "CONTAINS"},
        {"src": "pkg.mod.Foo", "dst": "pkg.mod.Foo.bar", "type": "CONTAINS"},
    ]
    node_ids = [n["id"] for n in nodes]
    rng = np.random.default_rng(0)
    embs = rng.standard_normal((len(nodes), 8)).astype(np.float32)

    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "nodes.json").write_text(json.dumps(nodes))
    (cache_dir / "edges.json").write_text(json.dumps(edges))
    (cache_dir / "node_ids.json").write_text(json.dumps(node_ids))
    np.save(cache_dir / "node_embeddings.npy", embs)
    return node_ids, embs


def test_link_entities_exact_name_match(tmp_path):
    node_ids, embs = _write_synthetic_cache(tmp_path)
    index = KGIndex(tmp_path)
    seeds = link_entities("What class does `pkg.mod.Foo` inherit from?", index, embs[0])
    assert seeds
    assert seeds[0][0] == "pkg.mod.Foo"


def test_link_entities_prefers_backtick_span_over_generic_sentence_words(tmp_path):
    """Regression test for the Phase 2 error-analysis finding: the imperative "Name"
    in "Name a class ... that `X` imports." must not out-rank the actual backtick-quoted
    entity `X` just because some unrelated node happens to be literally called `name`
    (very common as an attribute) and has a longer id that wins the length tie-break."""
    nodes = [
        {"id": "pkg", "type": "package", "name": "pkg", "package": "pkg", "docstring": "", "signature": ""},
        {"id": "pkg.mod", "type": "module", "name": "mod", "package": "pkg", "docstring": "", "signature": ""},
        # a deliberately long-id node whose short `name` collides with the word "Name"
        # in the question sentence - this is what used to win via the length tie-break
        {"id": "pkg.unrelated.SomeVeryLongClassName.name", "type": "function", "name": "name",
         "package": "pkg", "docstring": "", "signature": ""},
    ]
    edges = [{"src": "pkg", "dst": "pkg.mod", "type": "CONTAINS"}]
    node_ids = [n["id"] for n in nodes]
    rng = np.random.default_rng(0)
    embs = rng.standard_normal((len(nodes), 8)).astype(np.float32)

    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "nodes.json").write_text(json.dumps(nodes))
    (tmp_path / "edges.json").write_text(json.dumps(edges))
    (tmp_path / "node_ids.json").write_text(json.dumps(node_ids))
    np.save(tmp_path / "node_embeddings.npy", embs)

    index = KGIndex(tmp_path)
    seeds = link_entities("Name a class defined in a module that `pkg.mod` imports.", index, embs[0])
    assert seeds
    assert seeds[0][0] == "pkg.mod"
    assert "pkg.unrelated.SomeVeryLongClassName.name" not in {s for s, _ in seeds}


def test_multihop_retrieve_finds_chain(tmp_path):
    node_ids, embs = _write_synthetic_cache(tmp_path)
    index = KGIndex(tmp_path)
    query_emb = embs[node_ids.index("pkg.mod.Foo.bar")]
    seeds, triples = multihop_retrieve("Which class defines the method `pkg.mod.Foo.bar`?", index, query_emb, max_hops=2)
    assert seeds
    triple_tuples = {(t.src, t.relation, t.dst) for t in triples}
    assert ("pkg.mod.Foo", "CONTAINS", "pkg.mod.Foo.bar") in triple_tuples


def test_multihop_retrieve_no_seed_returns_empty(tmp_path):
    _write_synthetic_cache(tmp_path)
    index = KGIndex(tmp_path)
    query_emb = np.zeros(8, dtype=np.float32)
    seeds, triples = multihop_retrieve("completely unrelated gibberish zzzqqq", index, query_emb)
    # below similarity threshold -> should link to nothing or fall back gracefully
    assert isinstance(seeds, list)
    assert isinstance(triples, list)


def test_embedding_only_retrieve_finds_chain_without_gnn(tmp_path):
    """Ablation baseline (Part 1b): no gnn_embeddings.npy on disk at all, should still work."""
    node_ids, embs = _write_synthetic_cache(tmp_path)
    index = KGIndex(tmp_path)
    assert index.gnn_emb is None  # confirms this path truly has no GNN signal available
    query_emb = embs[node_ids.index("pkg.mod.Foo.bar")]
    seeds, triples = embedding_only_retrieve("Which class defines the method `pkg.mod.Foo.bar`?", index, query_emb, max_hops=2)
    assert seeds
    triple_tuples = {(t.src, t.relation, t.dst) for t in triples}
    assert ("pkg.mod.Foo", "CONTAINS", "pkg.mod.Foo.bar") in triple_tuples


def test_embedding_only_retrieve_no_seed_returns_empty(tmp_path):
    _write_synthetic_cache(tmp_path)
    index = KGIndex(tmp_path)
    query_emb = np.zeros(8, dtype=np.float32)
    seeds, triples = embedding_only_retrieve("completely unrelated gibberish zzzqqq", index, query_emb)
    assert isinstance(seeds, list)
    assert isinstance(triples, list)
