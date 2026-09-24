"""Phase 2, Part 2: a hand-curated entity-linking eval set targeting genuine ambiguity
in this KG - mentions that plausibly resolve to more than one real node, built from
real cross-package name collisions mined from data/cache/nodes.json (see the mining
query in the Phase 2 session notes / README), not invented after the fact.

Each example's `gold_entity` was determined by actually reading the colliding nodes'
docstrings and picking the one the `mention_context` sentence unambiguously describes.
`gold_entity: None` marks mentions that should resolve to NOTHING in this KG (mirrors
the Phase 1 hallucination case: a linker that forces a match here is the failure mode
this eval set exists to catch).
"""

from __future__ import annotations

import json
from pathlib import Path

EXAMPLES = [
    # --- cross-package CLASS name collisions (21) ---
    {"mention_context": "In Flask, what class is used by default to represent an incoming HTTP request?",
     "gold_entity": "flask.wrappers.Request", "distractors": ["werkzeug.wrappers.request.Request", "werkzeug.sansio.request.Request"]},
    {"mention_context": "In Werkzeug, which class represents an incoming WSGI HTTP request, with headers and body already taken from the WSGI environment?",
     "gold_entity": "werkzeug.wrappers.request.Request", "distractors": ["flask.wrappers.Request", "werkzeug.sansio.request.Request"]},
    {"mention_context": "What class does Flask use by default to represent an HTTP response?",
     "gold_entity": "flask.wrappers.Response", "distractors": ["werkzeug.wrappers.response.Response", "werkzeug.sansio.response.Response"]},
    {"mention_context": "In Werkzeug, which class represents an outgoing WSGI HTTP response with a body, status, and headers?",
     "gold_entity": "werkzeug.wrappers.response.Response", "distractors": ["flask.wrappers.Response", "werkzeug.sansio.response.Response"]},
    {"mention_context": "In Click, what class holds internal state relevant to a command-line script's execution?",
     "gold_entity": "click.core.Context", "distractors": ["jinja2.runtime.Context"]},
    {"mention_context": "In Jinja2, what class holds the variables available inside a template while it is being rendered?",
     "gold_entity": "jinja2.runtime.Context", "distractors": ["click.core.Context"]},
    {"mention_context": "In Click, what class declares a command-line parameter to be a file for reading or writing?",
     "gold_entity": "click.types.File", "distractors": ["werkzeug.sansio.multipart.File"]},
    {"mention_context": "In Click, what class lets you apply a type to each element of a tuple-valued command-line option?",
     "gold_entity": "click.types.Tuple", "distractors": ["jinja2.nodes.Tuple"]},
    {"mention_context": "In Jinja2's parser, what AST node represents a tuple used for loop unpacking or multiple call arguments?",
     "gold_entity": "jinja2.nodes.Tuple", "distractors": ["click.types.Tuple"]},
    {"mention_context": "In Flask's sansio layer, what class represents a blueprint - a collection of routes and other app-related functions that can be registered on an application?",
     "gold_entity": "flask.sansio.blueprints.Blueprint", "distractors": ["flask.blueprints.Blueprint"]},
    {"mention_context": "What is the core component of Jinja2 that contains important shared state like configuration, filters, tests, and global variables?",
     "gold_entity": "jinja2.environment.Environment", "distractors": ["flask.templating.Environment"]},
    {"mention_context": "In Flask, what class works like a regular Jinja environment but has extra knowledge of the Flask application's config and blueprints?",
     "gold_entity": "flask.templating.Environment", "distractors": ["jinja2.environment.Environment"]},
    {"mention_context": "In Werkzeug, what class lets you conveniently create a WSGI environment for testing purposes?",
     "gold_entity": "werkzeug.test.EnvironBuilder", "distractors": ["flask.testing.EnvironBuilder"]},
    {"mention_context": "In Flask's testing module, what class extends Werkzeug's EnvironBuilder to take its defaults from the Flask application?",
     "gold_entity": "flask.testing.EnvironBuilder", "distractors": ["werkzeug.test.EnvironBuilder"]},
    {"mention_context": "In Jinja2, what class represents a compiled template that is ready to be rendered?",
     "gold_entity": "jinja2.environment.Template", "distractors": ["jinja2.nodes.Template"]},
    {"mention_context": "In Jinja2's AST node module, what is the outermost node type representing an entire template?",
     "gold_entity": "jinja2.nodes.Template", "distractors": ["jinja2.environment.Template"]},
    {"mention_context": "In Jinja2, what exception is raised if a sandboxed template tries to do something insecure?",
     "gold_entity": "jinja2.exceptions.SecurityError", "distractors": ["werkzeug.exceptions.SecurityError"]},
    {"mention_context": "In Werkzeug, what HTTP exception class represents a generic security error?",
     "gold_entity": "werkzeug.exceptions.SecurityError", "distractors": ["jinja2.exceptions.SecurityError"]},
    {"mention_context": "In Jinja2's AST, what node type represents a macro definition with a name and an argument list?",
     "gold_entity": "jinja2.nodes.Macro", "distractors": ["jinja2.runtime.Macro"]},
    {"mention_context": "In Jinja2's runtime, what class wraps a macro function so it can be called at render time?",
     "gold_entity": "jinja2.runtime.Macro", "distractors": ["jinja2.nodes.Macro"]},
    {"mention_context": "In Werkzeug's URL routing matcher, what class represents a rule state?",
     "gold_entity": "werkzeug.routing.matcher.State", "distractors": ["werkzeug.sansio.multipart.State"]},

    # --- cross-package METHOD name collisions, disambiguated by holder class (15) ---
    {"mention_context": "In Jinja2's template runtime Context, what method looks up a variable by name and returns a default if it isn't found?",
     "gold_entity": "jinja2.runtime.Context.get", "distractors": ["flask.sansio.scaffold.Scaffold.get", "jinja2.utils.LRUCache.get"]},
    {"mention_context": "In Flask, what method on Scaffold is a shortcut for registering a route that only accepts GET requests?",
     "gold_entity": "flask.sansio.scaffold.Scaffold.get", "distractors": ["jinja2.runtime.Context.get"]},
    {"mention_context": "What method on Jinja2's LRUCache returns an item from the cache dict, or a default value if it's missing?",
     "gold_entity": "jinja2.utils.LRUCache.get", "distractors": ["jinja2.runtime.Context.get"]},
    {"mention_context": "In Click, what method on _LazyFile closes the underlying file no matter what?",
     "gold_entity": "click.utils._LazyFile.close", "distractors": ["click.core.Context.close"]},
    {"mention_context": "In Click's Context class, what method invokes all callbacks registered via call_on_close?",
     "gold_entity": "click.core.Context.close", "distractors": ["click.utils._LazyFile.close"]},
    {"mention_context": "In Click, what method on the base ParamType class converts a raw command-line value to its correct type?",
     "gold_entity": "click.types.ParamType.convert", "distractors": ["click.types.Choice.convert"]},
    {"mention_context": "In Click's Choice parameter type, what method normalizes a parsed value and finds its matching choice?",
     "gold_entity": "click.types.Choice.convert", "distractors": ["click.types.ParamType.convert"]},
    {"mention_context": "In Werkzeug, what method on the Headers class returns a copy of the headers?",
     "gold_entity": "werkzeug.datastructures.headers.Headers.copy", "distractors": ["jinja2.utils.LRUCache.copy"]},
    {"mention_context": "What method on Jinja2's LRUCache returns a shallow copy of the cache instance?",
     "gold_entity": "jinja2.utils.LRUCache.copy", "distractors": ["werkzeug.datastructures.headers.Headers.copy"]},
    {"mention_context": "In Jinja2's Environment class, what method parses template source code and returns its abstract syntax tree?",
     "gold_entity": "jinja2.environment.Environment.parse", "distractors": []},
    {"mention_context": "In Werkzeug's Headers class, what method removes and returns a header by key or index?",
     "gold_entity": "werkzeug.datastructures.headers.Headers.pop", "distractors": ["flask.ctx.AppContext.pop"]},
    {"mention_context": "In Flask, what method on AppContext pops the context so it's no longer the active one?",
     "gold_entity": "flask.ctx.AppContext.pop", "distractors": ["werkzeug.datastructures.headers.Headers.pop"]},
    {"mention_context": "In Werkzeug's Headers class, what method replaces the headers in this object with items from another headers object?",
     "gold_entity": "werkzeug.datastructures.headers.Headers.update", "distractors": []},
    {"mention_context": "In Werkzeug, what method on the Headers class clears all headers?",
     "gold_entity": "werkzeug.datastructures.headers.Headers.clear", "distractors": ["jinja2.utils.LRUCache.clear"]},
    {"mention_context": "What method on Jinja2's LRUCache clears the entire cache?",
     "gold_entity": "jinja2.utils.LRUCache.clear", "distractors": ["werkzeug.datastructures.headers.Headers.clear"]},

    # --- abbreviations resolved by context (2) ---
    {"mention_context": "What does `ctx` typically refer to when it's the first argument to a Click command callback?",
     "gold_entity": "click.core.Context", "distractors": ["jinja2.runtime.Context"]},
    {"mention_context": "In Jinja2, what is the `env` object commonly used to render templates called as a class?",
     "gold_entity": "jinja2.environment.Environment", "distractors": ["flask.templating.Environment"]},

    # --- mentions that should resolve to NOTHING in this KG (2) - the failure mode ---
    # this eval set exists to catch: does the linker force a match to some unrelated
    # real node instead of correctly reporting no confident match?
    {"mention_context": "What does `requests.sessions.Session` inherit from?",
     "gold_entity": None, "distractors": []},
    {"mention_context": "What does the class `SmartTemplate` in a Jinja2 blockchain extension module do?",
     "gold_entity": None, "distractors": []},
]


def build() -> list[dict]:
    for ex in EXAMPLES:
        ex["ambiguity_type"] = (
            "out_of_kg" if ex["gold_entity"] is None
            else "abbreviation" if ex is EXAMPLES[-4] or ex is EXAMPLES[-3]
            else "cross_package_method_name" if "." in ex["gold_entity"].rsplit(".", 1)[0] and ex["gold_entity"].split(".")[-1].islower()
            else "cross_package_class_name"
        )
    return EXAMPLES


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    examples = build()
    out_path = repo_root / "data" / "linking_eval_set.json"
    out_path.write_text(json.dumps(examples, indent=2))
    print(f"wrote {len(examples)} linking-ambiguity examples -> {out_path}")
