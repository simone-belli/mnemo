"""SQLite schema and the small queries the rest of the app needs.

One file (`data/mnemo.db`). Every request opens its own connection; WAL mode lets the
threadpool read while another request writes.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes(
  id INTEGER PRIMARY KEY, parent_id INTEGER NULL, name TEXT NOT NULL, description TEXT NOT NULL,
  depth INTEGER NOT NULL, embedding BLOB, created_at TEXT NOT NULL,
  source TEXT NOT NULL                      -- seed | lazy | quiz
);
CREATE INDEX IF NOT EXISTS nodes_parent ON nodes(parent_id);

CREATE TABLE IF NOT EXISTS mastery(
  node_id INTEGER, mode TEXT, alpha REAL NOT NULL, beta REAL NOT NULL, last_seen TEXT NULL,
  PRIMARY KEY(node_id, mode)
);

CREATE TABLE IF NOT EXISTS interactions(
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL,
  kind TEXT NOT NULL,                       -- recall | explain | feedback | quiz | rating (legacy)
  query TEXT, node_id INTEGER, mode TEXT,
  tag_dist TEXT,                            -- JSON
  response_md TEXT, audio_script TEXT, audio_path TEXT,
  feedback TEXT NULL,                       -- difficulty:1..5 or repeat (older: the button), or the quiz score
  diagnosis TEXT NULL,                      -- JSON
  ref_id INTEGER NULL,                      -- feedback: the recall it is about; quiz: the quiz item
  deltas TEXT NULL                          -- JSON list of mastery changes applied (for the debug drawer)
);

CREATE TABLE IF NOT EXISTS preferences(branch_node_id INTEGER PRIMARY KEY, depth_pref REAL NOT NULL);

CREATE TABLE IF NOT EXISTS quiz_items(
  id INTEGER PRIMARY KEY, node_id INTEGER, mode TEXT, difficulty REAL,
  question TEXT, options TEXT NULL, answer TEXT NULL,
  user_answer TEXT, score REAL, ts TEXT
);

CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_ts(ts: str | None) -> datetime | None:
    return datetime.fromisoformat(ts) if ts else None


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


# ---------------------------------------------------------------- meta

def get_meta(conn, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn, key: str, value: str) -> None:
    conn.execute("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                 (key, value))


# ---------------------------------------------------------------- nodes

def to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob: bytes | None) -> np.ndarray | None:
    return None if blob is None else np.frombuffer(blob, dtype=np.float32)


def insert_node(conn, name: str, description: str, parent_id: int | None, source: str,
                embedding: np.ndarray | None = None, ts: str | None = None) -> int:
    depth = 0 if parent_id is None else get_node(conn, parent_id)["depth"] + 1
    cur = conn.execute(
        "INSERT INTO nodes(parent_id, name, description, depth, embedding, created_at, source) VALUES(?,?,?,?,?,?,?)",
        (parent_id, name, description, depth, None if embedding is None else to_blob(embedding), ts or now_iso(), source))
    return cur.lastrowid


def get_node(conn, node_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()


def children(conn, node_id: int | None) -> list[sqlite3.Row]:
    if node_id is None:
        return conn.execute("SELECT * FROM nodes WHERE parent_id IS NULL ORDER BY id").fetchall()
    return conn.execute("SELECT * FROM nodes WHERE parent_id=? ORDER BY id", (node_id,)).fetchall()


def all_nodes(conn) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM nodes ORDER BY id").fetchall()


def path(conn, node_id: int) -> list[sqlite3.Row]:
    """Root first, the node itself last."""
    out = []
    node = get_node(conn, node_id)
    while node is not None:
        out.append(node)
        node = get_node(conn, node["parent_id"]) if node["parent_id"] is not None else None
    return out[::-1]


def find_child_by_name(conn, parent_id: int, name: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM nodes WHERE parent_id=? AND lower(name)=lower(?)",
                        (parent_id, name.strip())).fetchone()


# ---------------------------------------------------------------- mastery

def get_cells(conn) -> dict[tuple[int, str], tuple[float, float, str | None]]:
    return {(r["node_id"], r["mode"]): (r["alpha"], r["beta"], r["last_seen"])
            for r in conn.execute("SELECT * FROM mastery")}


def get_cell(conn, node_id: int, mode: str) -> tuple[float, float, str | None] | None:
    r = conn.execute("SELECT * FROM mastery WHERE node_id=? AND mode=?", (node_id, mode)).fetchone()
    return (r["alpha"], r["beta"], r["last_seen"]) if r else None


def put_cell(conn, node_id: int, mode: str, alpha: float, beta: float, last_seen: str | None) -> None:
    conn.execute(
        "INSERT INTO mastery(node_id, mode, alpha, beta, last_seen) VALUES(?,?,?,?,?) "
        "ON CONFLICT(node_id, mode) DO UPDATE SET alpha=excluded.alpha, beta=excluded.beta, last_seen=excluded.last_seen",
        (node_id, mode, alpha, beta, last_seen))


# ---------------------------------------------------------------- interactions

def log_interaction(conn, kind: str, **fields) -> int:
    fields = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in fields.items()}
    fields.setdefault("ts", now_iso())
    cols = ", ".join(["kind", *fields])
    marks = ", ".join("?" * (len(fields) + 1))
    cur = conn.execute(f"INSERT INTO interactions({cols}) VALUES({marks})", (kind, *fields.values()))
    return cur.lastrowid


def get_interaction(conn, interaction_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM interactions WHERE id=?", (interaction_id,)).fetchone()


def update_interaction(conn, interaction_id: int, **fields) -> None:
    fields = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in fields.items()}
    sets = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE interactions SET {sets} WHERE id=?", (*fields.values(), interaction_id))


# ---------------------------------------------------------------- preferences

def get_depth_pref(conn, branch_node_id: int) -> float:
    r = conn.execute("SELECT depth_pref FROM preferences WHERE branch_node_id=?", (branch_node_id,)).fetchone()
    return r["depth_pref"] if r else 0.0


def put_depth_pref(conn, branch_node_id: int, value: float) -> None:
    conn.execute("INSERT INTO preferences(branch_node_id, depth_pref) VALUES(?, ?) "
                 "ON CONFLICT(branch_node_id) DO UPDATE SET depth_pref=excluded.depth_pref", (branch_node_id, value))
