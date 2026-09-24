from kg_grounded_qa.learned_linker import _mention_variants, build_training_pairs


def test_mention_variants_includes_name_and_full_id():
    node = {"id": "flask.app.Flask.run", "name": "run", "type": "function", "docstring": ""}
    variants = _mention_variants(node)
    assert "run" in variants
    assert "flask.app.Flask.run" in variants


def test_mention_variants_includes_truncated_suffixes():
    node = {"id": "flask.app.Flask.run", "name": "run", "type": "function", "docstring": ""}
    variants = _mention_variants(node)
    assert "Flask.run" in variants
    assert "app.Flask.run" in variants


def test_mention_variants_includes_templated_paraphrase():
    node = {"id": "flask.app.Flask", "name": "Flask", "type": "class", "docstring": ""}
    variants = _mention_variants(node)
    assert "the class Flask" in variants


def test_mention_variants_includes_docstring_first_sentence():
    node = {"id": "pkg.mod.Foo", "name": "Foo", "type": "class", "docstring": "Represents a widget. More detail here."}
    variants = _mention_variants(node)
    assert "Represents a widget" in variants


def test_mention_variants_no_empty_strings():
    node = {"id": "pkg", "name": "pkg", "type": "package", "docstring": ""}
    variants = _mention_variants(node)
    assert "" not in variants


def test_build_training_pairs_covers_every_node():
    nodes = [
        {"id": "a.b", "name": "b", "type": "module", "docstring": ""},
        {"id": "a.b.C", "name": "C", "type": "class", "docstring": ""},
    ]
    pairs = build_training_pairs(nodes)
    mentions = {m for m, _ in pairs}
    assert "b" in mentions
    assert "C" in mentions
    assert len(pairs) >= len(nodes)


def test_build_training_pairs_positive_pair_uses_node_text():
    from kg_grounded_qa.embeddings import node_text

    node = {"id": "a.b", "name": "b", "type": "module", "docstring": "", "signature": ""}
    pairs = build_training_pairs([node])
    expected_entity_text = node_text(node)
    assert all(entity_text == expected_entity_text for _, entity_text in pairs)


def test_build_training_pairs_respects_max_pairs():
    nodes = [{"id": f"pkg.mod{i}", "name": f"mod{i}", "type": "module", "docstring": ""} for i in range(20)]
    pairs = build_training_pairs(nodes, max_pairs=5)
    assert len(pairs) == 5
