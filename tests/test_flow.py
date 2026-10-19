"""End-to-end API tests with fake Claude/ElevenLabs (SPEC §12 acceptance criteria, where testable offline)."""
import pytest

from app import db, knowledge, quiz
from app.fakes import QUIZ_BANK
from app.main import create_app


def recall(client, query, **kw):
    r = client.post("/api/recall", json={"query": query, "session_id": "t", **kw})
    assert r.status_code == 200, r.text
    return r.json()


def node_by_name(svc, name):
    with svc.db() as conn:
        return conn.execute("SELECT * FROM nodes WHERE name=?", (name,)).fetchone()


def test_seed_tree(svc, client):
    with svc.db() as conn:
        roots = db.children(conn, None)
        assert [r["name"] for r in roots] == ["formal", "nature", "mind", "society", "artefacts", "ultimate"]
        for r in roots:
            assert 8 <= len(db.children(conn, r["id"])) <= 12
        assert all(n["embedding"] is not None for n in db.all_nodes(conn))


def test_recall_payload_and_audio_cache(svc, client):
    r = recall(client, "remind me about statistical mechanics")
    for key in ("text_md", "audio_script", "audio_url", "timings", "node_path", "mode", "next_steps", "interaction_id"):
        assert key in r
    assert r["node_path"][0] in ("nature", "formal", "mind", "society", "artefacts", "ultimate")
    assert len(r["timings"]) == len(r["audio_script"])
    assert "$" not in r["audio_script"] and "*" not in r["audio_script"]
    assert 2 <= len(r["next_steps"]) <= 3
    audio = client.get(r["audio_url"])
    assert audio.status_code == 200 and audio.content[:4] == b"RIFF"
    # The same recap again: the audio comes from the cache, no new synthesis.
    before = svc.tts.requests
    r2 = recall(client, "remind me about statistical mechanics")
    assert r2["audio_cached"] and svc.tts.requests == before


def test_read_mode_skips_audio_until_asked(svc, client):
    r = recall(client, "remind me about statistical mechanics", audio=False)
    assert r["audio_script"] is None and r["audio_url"] is None
    a = client.post("/api/audio", json={"interaction_id": r["interaction_id"]}).json()
    assert a["audio_script"] and a["audio_url"]


def test_knew_that_twice_makes_next_recap_more_advanced(svc, client):
    q = "remind me about statistical mechanics"
    first = recall(client, q)
    mode = first["mode"]
    assert first["debug"]["profile"]["node"]["mastery_by_mode"][mode]["mean"] == 0.5
    assert "Level: intermediate" in first["text_md"]
    for _ in range(2):
        fb = client.post("/api/feedback", json={"interaction_id": first["interaction_id"], "signal": "knew_that"}).json()
        assert fb["deltas"][0]["d_alpha"] == 1.0
    assert fb["deltas"][0]["after"] == pytest.approx(0.75)
    again = recall(client, q)
    assert again["node_id"] == first["node_id"]
    assert again["debug"]["profile"]["node"]["mastery_by_mode"][mode]["mean"] == 0.75
    assert '"mean": 0.75' in again["debug"]["user_prompt"]
    assert "Level: advanced" in again["text_md"]


def test_lost_me_logs_prerequisite_and_next_recap_covers_it(svc, client):
    q = "remind me about Renaissance painting"
    r = recall(client, q)
    fb = client.post("/api/feedback", json={"interaction_id": r["interaction_id"], "signal": "lost_me",
                                            "note": "linear perspective"}).json()
    d = fb["diagnosis"]
    assert d["kind"] == "missing_prerequisite" and d["term"] == "linear perspective"
    assert d["target_node_id"] and fb["deltas"][0]["node_id"] == d["target_node_id"]
    assert fb["deltas"][0]["d_beta"] == 1.0
    with svc.db() as conn:
        row = conn.execute("SELECT * FROM interactions WHERE kind='feedback'").fetchone()
        assert "linear perspective" in row["diagnosis"]
    again = recall(client, q)
    assert again["debug"]["profile"]["recent_gaps"][0]["term"] == "linear perspective"
    assert again["text_md"].startswith("**First, linear perspective.**")


def test_simpler_and_deeper_move_depth_preference(svc, client):
    r = recall(client, "remind me about statistical mechanics")
    for _ in range(2):
        fb = client.post("/api/feedback", json={"interaction_id": r["interaction_id"], "signal": "deeper"}).json()
    assert fb["depth_preference"] == 0.5
    again = recall(client, "remind me about statistical mechanics")
    assert "go deeper" in again["debug"]["user_prompt"]
    fb = client.post("/api/feedback", json={"interaction_id": r["interaction_id"], "signal": "simpler"}).json()
    assert fb["depth_preference"] == 0.25 and fb["deltas"][0]["d_beta"] == 0.5


def test_bad_feedback(client):
    r = recall(client, "remind me about statistical mechanics")
    assert client.post("/api/feedback", json={"interaction_id": r["interaction_id"], "signal": "meh"}).status_code == 400
    assert client.post("/api/feedback", json={"interaction_id": 999, "signal": "deeper"}).status_code == 404


def test_vague_query_shallow_specific_query_deep(svc, client):
    vague = recall(client, "tell me about art")
    assert len(vague["node_path"]) <= 2 and not vague["debug"]["created_nodes"]
    # The hash embedder only matches shared words, so the query names its field.
    specific = recall(client, "Brunelleschi's perspective experiment in visual art painting")
    assert len(specific["node_path"]) > len(vague["node_path"])
    assert specific["debug"]["created_nodes"]
    with svc.db() as conn:
        node = db.get_node(conn, specific["node_id"])
        assert node["source"] == "lazy" and "brunelleschi" in node["name"]
        # Correlation prior: two pseudo-counts at the parent's mean.
        a, b, _ = db.get_cell(conn, node["id"], "descriptive")
        assert a + b == pytest.approx(2.0)


def test_next_steps_follow_the_frontier(svc, client):
    client.post("/api/quiz/rating", json={"ratings": {"nature": 5}})
    nature = node_by_name(svc, "nature")
    steps = client.get(f"/api/next-steps?node_id={nature['id']}").json()["next_steps"]
    assert len(steps) == 3
    assert all(s["path"][0] == "nature" and s["score"] > 0 for s in steps)


def correct_option(svc, item_id: int) -> str:
    with svc.db() as conn:
        return conn.execute("SELECT answer FROM quiz_items WHERE id=?", (item_id,)).fetchone()["answer"]


def test_quiz_flow(svc, client):
    client.post("/api/quiz/rating", json={"ratings": {"formal": 4, "nature": 5, "artefacts": 2}})
    item = client.get("/api/quiz/next").json()
    assert item["options"] and len(item["options"]) == 4 and item["progress"] == {"answered": 0, "target": 20}
    assert "true one" not in " ".join(item["options"]).lower()      # a real question from the offline bank
    assert client.get("/api/quiz/next").json()["item_id"] == item["item_id"]       # unanswered item is reused
    right = correct_option(svc, item["item_id"])
    r = client.post("/api/quiz/answer", json={"item_id": item["item_id"], "answer": right}).json()
    assert r["score"] == 1.0 and r["deltas"][0]["d_alpha"] > 0
    assert len(r["deltas"]) > 1                                       # strong answer: siblings get a boost
    assert all(d["d_alpha"] == 0.2 for d in r["deltas"][1:])
    assert client.post("/api/quiz/answer", json={"item_id": item["item_id"], "answer": right}).status_code == 409
    nxt = r["next"]
    assert nxt["progress"]["answered"] == 1 and nxt["item_id"] != item["item_id"]
    wrong = str((int(correct_option(svc, nxt["item_id"])) + 1) % 4)
    r = client.post("/api/quiz/answer", json={"item_id": nxt["item_id"], "answer": wrong}).json()
    assert r["score"] == 0.0 and r["deltas"][0]["d_beta"] > 0
    free = r["next"]                                                  # every third item is free text
    assert free["options"] is None
    model_answer = correct_option(svc, free["item_id"])
    r = client.post("/api/quiz/answer", json={"item_id": free["item_id"], "answer": model_answer}).json()
    assert r["score"] == 1.0 and r["correct"] == model_answer


def test_offline_bank_covers_the_first_quiz_level(svc):
    with svc.db() as conn:
        assert {n["name"] for n in db.all_nodes(conn) if n["depth"] == 1} <= set(QUIZ_BANK)
    for entry in QUIZ_BANK.values():
        assert len(entry["choice"]) == 5 and len(entry["short"]) == 3


def test_quiz_shuffles_options_and_drops_stale_items(svc, client):
    """The right option isn't always first, and an unanswered item doesn't outlive a restart."""
    with svc.db() as conn:
        node = node_by_name(svc, "music")["id"]
        firsts = set()
        for _ in range(12):
            item_id = quiz.generate_item(conn, svc.settings, svc.llm, node, "descriptive", 0.5, multiple_choice=True)
            firsts.add(correct_option(svc, item_id))
        assert len(firsts) > 1
    stale = client.get("/api/quiz/next").json()["item_id"]
    create_app(svc)
    assert client.get("/api/quiz/next").json()["item_id"] != stale


def test_map_endpoint(svc, client):
    roots = client.get("/api/map").json()["children"]
    assert len(roots) == 6
    node = client.get(f"/api/map?node_id={roots[1]['id']}").json()
    assert node["node"]["name"] == "nature" and node["children"]
    assert set(node["node"]["cells"]) == {"descriptive", "explanatory", "normative", "practical", "expressive"}


def test_map_rebuilds_from_interactions(svc, client):
    """SPEC §12: the whole map can be rebuilt by replaying `interactions`."""
    client.post("/api/quiz/rating", json={"ratings": {"nature": 4, "society": 2}})
    r = recall(client, "remind me about statistical mechanics")
    for signal in ("knew_that", "deeper", "repeat", "simpler"):
        client.post("/api/feedback", json={"interaction_id": r["interaction_id"], "signal": signal})
    r2 = recall(client, "Brunelleschi's perspective experiment")
    client.post("/api/feedback", json={"interaction_id": r2["interaction_id"], "signal": "lost_me", "note": "vanishing point"})
    item = client.get("/api/quiz/next").json()
    client.post("/api/quiz/answer", json={"item_id": item["item_id"], "answer": "0"})

    with svc.db() as conn:
        live = {k: (round(a, 6), round(b, 6), ts) for k, (a, b, ts) in db.get_cells(conn).items()}
        prefs = {r["branch_node_id"]: r["depth_pref"] for r in conn.execute("SELECT * FROM preferences")}
    assert client.post("/api/replay").status_code == 200
    with svc.db() as conn:
        rebuilt = {k: (round(a, 6), round(b, 6), ts) for k, (a, b, ts) in db.get_cells(conn).items()}
        prefs2 = {r["branch_node_id"]: r["depth_pref"] for r in conn.execute("SELECT * FROM preferences")}
    assert rebuilt == live
    assert prefs2 == prefs


def test_profile_shape(svc, client):
    r = recall(client, "remind me about statistical mechanics")
    p = r["debug"]["profile"]
    assert set(p) >= {"topic", "mode", "node", "parent", "siblings", "children", "recent_gaps", "depth_preference"}
    with svc.db() as conn:
        assert knowledge.branch_of(conn, r["node_id"]) in [n["id"] for n in db.path(conn, r["node_id"])]


def test_audio_route_rejects_odd_names(client):
    assert client.get("/audio/../mnemo.db").status_code == 404
    assert client.get("/audio/abcdef12.mp3").status_code == 404
