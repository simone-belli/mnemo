"""The domain tree (SPEC §3.2): seeding, embeddings and lazy refinement.

`seed_taxonomy.json` holds the six roots and their fields; edit it by hand freely (it is only
read when the database has no nodes). Deeper nodes are created on demand by `refine()`.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from . import db, knowledge
from .config import Settings
from .embed import Embedder
from .knowledge import iso
from .llm import obj

SEED_FILE = Path(__file__).with_name("seed_taxonomy.json")


def node_text(name: str, description: str) -> str:
    return f"{name}: {description}"


def seed(conn, embedder: Embedder, path: Path = SEED_FILE) -> int:
    """Create the seed tree if the database is empty. Returns the number of nodes created."""
    if db.all_nodes(conn):
        return 0
    count = 0

    def add(item: dict, parent_id: int | None) -> None:
        nonlocal count
        nid = db.insert_node(conn, item["name"], item["description"], parent_id, "seed")
        count += 1
        for child in item.get("children", []):
            add(child, nid)

    for root in json.loads(path.read_text()):
        add(root, None)
    ensure_embeddings(conn, embedder)
    conn.commit()
    return count


def ensure_embeddings(conn, embedder: Embedder) -> None:
    """Embed nodes that have no embedding; re-embed everything if the embedding model changed."""
    if db.get_meta(conn, "embed_model") != embedder.name:
        conn.execute("UPDATE nodes SET embedding=NULL")
        db.set_meta(conn, "embed_model", embedder.name)
    rows = conn.execute("SELECT id, name, description FROM nodes WHERE embedding IS NULL").fetchall()
    if not rows:
        return
    vecs = embedder.embed([node_text(r["name"], r["description"]) for r in rows])
    for r, v in zip(rows, vecs):
        conn.execute("UPDATE nodes SET embedding=? WHERE id=?", (db.to_blob(v), r["id"]))
    conn.commit()


def create_child(conn, settings: Settings, embedder: Embedder, parent_id: int, name: str, description: str,
                 source: str, now: datetime) -> int:
    """Add one child (or return the existing one with that name), embedded and with its prior."""
    existing = db.find_child_by_name(conn, parent_id, name)
    if existing:
        return existing["id"]
    vec = embedder.embed([node_text(name, description)])[0]
    nid = db.insert_node(conn, name, description, parent_id, source, vec, ts=iso(now))
    knowledge.init_child_cells(conn, settings, nid, now)
    return nid


# ---------------------------------------------------------------- lazy refinement

REFINE_SYSTEM = """You maintain a hierarchical map of human knowledge for one person.
A node B is a CHILD of node A when B is subsumed by A: texts about B almost always involve A,
but texts about A often do not involve B. Children should be distinct subfields at the next
natural level of detail (not examples, not the query restated, not overlapping)."""

REFINE_SCHEMA = obj({
    "children": {"type": "array", "items": obj({"name": {"type": "string"}, "description": {"type": "string"}})},
    "match_index": {"type": "integer"},
    "still_more_specific": {"type": "boolean"},
})


def propose_children(llm, settings: Settings, conn, node_id: int, query: str | None) -> dict:
    path = " › ".join(n["name"] for n in db.path(conn, node_id))
    node = db.get_node(conn, node_id)
    existing = [c["name"] for c in db.children(conn, node_id)]
    prompt = (
        f"Node: {path}\nDescription: {node['description']}\n"
        f"Existing children: {', '.join(existing) or '(none)'}\n\n"
        "Propose 3 to 8 children of this node, in lowercase short names with a one-line description. "
        "Do not repeat existing children; you may include an existing child's exact name if it is the one that fits "
        "the query.\n"
    )
    if query:
        prompt += (
            f"\nQuery: {query!r}\nOne child must be the natural home of this query (as general as a sibling, "
            "not the query itself, unless the query is already a natural subfield). Give its index in "
            "`match_index`, and set `still_more_specific` if the query is still clearly narrower than that child."
        )
    else:
        prompt += "\nSet `match_index` to -1 and `still_more_specific` to false."
    return llm.json("refine", REFINE_SYSTEM, prompt, REFINE_SCHEMA, settings.fast_model)


def refine(conn, settings: Settings, llm, embedder: Embedder, node_id: int, query: str | None,
           now: datetime, source: str = "lazy") -> tuple[int, bool, list[int]]:
    """Create children of `node_id` (adaptive mesh refinement for knowledge).

    Returns (the child matching the query, or `node_id` itself; whether the query is still
    more specific than that; the ids of the children)."""
    out = propose_children(llm, settings, conn, node_id, query)
    ids = [create_child(conn, settings, embedder, node_id, c["name"].strip().lower(), c["description"].strip(),
                        source, now)
           for c in out["children"][:8] if c["name"].strip()]
    conn.commit()
    i = out.get("match_index", -1)
    if query and 0 <= i < len(ids):
        return ids[i], bool(out.get("still_more_specific")), ids
    return node_id, False, ids


def generate_seed(llm, settings: Settings, path: Path = SEED_FILE) -> None:
    """One-off helper: ask the LLM for the second level of the six roots and save it for hand-editing.

    python -c "from app.main import services; from app.taxonomy import generate_seed; s = services(); generate_seed(s.llm, s.settings)"
    """
    roots = [{k: r[k] for k in ("name", "description")} for r in json.loads(path.read_text())]
    schema = obj({"children": {"type": "array",
                               "items": obj({"name": {"type": "string"}, "description": {"type": "string"}})}})
    for r in roots:
        prompt = (f"Root domain: {r['name']} ({r['description']}). Propose 8 to 12 second-level fields, "
                  "lowercase short names with a one-line description, covering the domain without overlaps.")
        r["children"] = llm.json("seed", REFINE_SYSTEM, prompt, schema, settings.fast_model)["children"]
    path.write_text(json.dumps(roots, indent=2, ensure_ascii=False) + "\n")
