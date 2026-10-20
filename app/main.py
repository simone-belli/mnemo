"""FastAPI routes (SPEC §9). Run with: uvicorn app.main:app --reload"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db, feedback, knowledge, quiz, recap, taxonomy
from .config import ROOT, Settings
from .embed import Embedder, make_embedder
from .llm import LLM, LLMError
from .tagging import Tagger
from .tts import AudioCache, ElevenLabsTTS, TTSError

log = logging.getLogger("mnemo")
STATIC = ROOT / "static"


@dataclass
class Services:
    settings: Settings
    llm: object
    embedder: Embedder
    tts: object
    tagger: Tagger

    @contextmanager
    def db(self):
        conn = db.connect(self.settings.db_path)
        try:
            yield conn
        finally:
            conn.close()


def services(settings: Settings | None = None) -> Services:
    s = settings or Settings()
    cache = AudioCache(s.audio_dir)
    if s.fake:
        from .fakes import FakeLLM, FakeTTS
        llm, embedder, tts = FakeLLM(), make_embedder("hash"), FakeTTS(cache)
    else:
        llm = LLM(s.anthropic_api_key, s.anthropic_workspace_id)
        embedder = make_embedder(s.embed_model)
        tts = ElevenLabsTTS(s.elevenlabs_api_key, s.elevenlabs_voice_id, s.elevenlabs_model, cache)
    return Services(s, llm, embedder, tts, Tagger(s, llm, embedder))


# ---------------------------------------------------------------- request bodies

class RecallIn(BaseModel):
    query: str
    session_id: str | None = None
    node_id: int | None = None       # set when a next-step chip is tapped
    audio: bool = True               # False in Read mode: skip the script and TTS until asked


class AudioIn(BaseModel):
    interaction_id: int


class FeedbackIn(BaseModel):
    interaction_id: int
    signal: str
    note: str | None = None


class AnswerIn(BaseModel):
    item_id: int
    answer: str = ""
    dont_know: bool = False


# ---------------------------------------------------------------- the app

def create_app(svc: Services | None = None) -> FastAPI:
    svc = svc or services()
    s = svc.settings
    with svc.db() as conn:
        taxonomy.seed(conn, svc.embedder)
        taxonomy.ensure_embeddings(conn, svc.embedder)
        quiz.discard_pending(conn)

    app = FastAPI(title="Mnemo")
    app.state.svc = svc
    pool = ThreadPoolExecutor(max_workers=4)

    def fail(e: Exception):
        log.exception("request failed")
        raise HTTPException(502, str(e)) from e

    # ------------------------------------------------------------ recall (SPEC §5)

    def next_steps(node_id: int, query_vec, now) -> list[dict]:
        """Frontier nodes near the query; topped up with the least-known children of the node
        (created on demand) when the frontier is thin, e.g. before the map knows much."""
        with svc.db() as conn:
            steps = knowledge.frontier(conn, s, query_vec, node_id, now, k=3)
            if len(steps) < 2:
                kids = db.children(conn, node_id)
                if not kids:
                    try:
                        taxonomy.refine(conn, s, svc.llm, svc.embedder, node_id, None, now)
                    except LLMError:
                        log.warning("could not create children for next steps", exc_info=True)
                    kids = db.children(conn, node_id)
                view = knowledge.MapView(conn, s, now)
                seen = {st["node_id"] for st in steps}
                for k in sorted(kids, key=lambda k: view.node(k["id"])[0]):
                    if len(steps) >= 3:
                        break
                    if k["id"] not in seen:
                        steps.append({"node_id": k["id"], "name": k["name"], "score": None})
            for st in steps:
                st["path"] = [n["name"] for n in db.path(conn, st["node_id"])]
            return steps

    def make_audio(conn, interaction_id: int, text_md: str) -> dict:
        script = recap.write_script(svc.llm, s, text_md)
        out = {"audio_script": script, "audio_url": None, "timings": None, "audio_error": None}
        if svc.tts.available:
            try:
                a = svc.tts.synthesize(script)
                out |= {"audio_url": a["url"], "timings": a["starts"], "audio_cached": a["cached"]}
            except TTSError as e:
                out["audio_error"] = str(e)
        else:
            out["audio_error"] = "No ElevenLabs key: using the browser's voice."
        db.update_interaction(conn, interaction_id, audio_script=script, audio_path=out["audio_url"])
        conn.commit()
        return out

    @app.post("/api/recall")
    def recall(body: RecallIn):
        query = body.query.strip()
        if not query:
            raise HTTPException(400, "empty query")
        now = knowledge.utcnow()
        with svc.db() as conn:
            try:
                tag = svc.tagger.tag(conn, query, now, body.session_id, forced_node=body.node_id)
                conn.commit()
                prof = knowledge.profile(conn, s, tag.node_id, tag.mode, now)
                f_steps = pool.submit(next_steps, tag.node_id, tag.query_vec, now)
                system, text_md = recap.write_recap(svc.llm, s, query, prof)
            except LLMError as e:
                fail(e)
            ev = {"kind": "recall", "query": query, "node_id": tag.node_id, "mode": tag.mode,
                  "tag_dist": tag.dist, "response_md": text_md, "ts": knowledge.iso(now)}
            deltas = knowledge.apply_event(conn, s, ev, now)
            iid = db.log_interaction(conn, **ev)
            conn.commit()
            out = {"interaction_id": iid, "text_md": text_md, "audio_script": None, "audio_url": None,
                   "timings": None, "node_id": tag.node_id, "node_path": tag.path, "mode": tag.mode}
            if body.audio:
                try:
                    out |= make_audio(conn, iid, text_md)
                except LLMError as e:
                    out["audio_error"] = str(e)
            try:
                out["next_steps"] = f_steps.result(timeout=60)
            except Exception:                         # suggestions are a nicety; never fail the recall
                log.warning("next steps failed", exc_info=True)
                out["next_steps"] = []
            out["debug"] = {"tag_dist": tag.dist, "tagging": tag.debug, "created_nodes": tag.created,
                            "profile": prof, "system_prompt": system,
                            "user_prompt": recap.recap_prompt(s, query, prof), "deltas": deltas}
            return out

    @app.post("/api/audio")
    def audio_for(body: AudioIn):
        with svc.db() as conn:
            rec = db.get_interaction(conn, body.interaction_id)
            if rec is None or rec["kind"] != "recall":
                raise HTTPException(404, "no such recall")
            try:
                return make_audio(conn, rec["id"], rec["response_md"])
            except LLMError as e:
                fail(e)

    # ------------------------------------------------------------ feedback (SPEC §6)

    @app.post("/api/feedback")
    def give_feedback(body: FeedbackIn):
        with svc.db() as conn:
            try:
                return feedback.handle(conn, s, svc.llm, svc.tagger, body.interaction_id, body.signal, body.note,
                                       knowledge.utcnow())
            except KeyError:
                raise HTTPException(404, "no such recall")
            except ValueError as e:
                raise HTTPException(400, str(e))
            except LLMError as e:
                fail(e)

    # ------------------------------------------------------------ quiz (SPEC §7)

    @app.get("/api/quiz/next")
    def quiz_next():
        with svc.db() as conn:
            try:
                return quiz.next_item(conn, s, svc.llm, svc.embedder, knowledge.utcnow())
            except LLMError as e:
                fail(e)

    @app.post("/api/quiz/answer")
    def quiz_answer(body: AnswerIn):
        with svc.db() as conn:
            try:
                out = quiz.answer(conn, s, svc.llm, body.item_id, body.answer, knowledge.utcnow(),
                                  dont_know=body.dont_know)
                done = quiz.round_answered(conn) >= s.quiz_target
                out["next"] = None if done else quiz.next_item(conn, s, svc.llm, svc.embedder, knowledge.utcnow())
                return out
            except KeyError:
                raise HTTPException(404, "no such item")
            except ValueError as e:
                raise HTTPException(409, str(e))
            except LLMError as e:
                fail(e)

    @app.post("/api/quiz/start")
    def quiz_start():
        with svc.db() as conn:
            quiz.start_round(conn)
            return {"ok": True}

    # ------------------------------------------------------------ map and next steps

    @app.get("/api/state")
    def state():
        with svc.db() as conn:
            count = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
            return {"recalls": count("SELECT count(*) FROM interactions WHERE kind='recall'"),
                    "quiz_answered": count("SELECT count(*) FROM quiz_items WHERE score IS NOT NULL"),
                    "quiz_target": s.quiz_target, "tts": svc.tts.name if svc.tts.available else None}

    @app.get("/api/map")
    def map_node(node_id: int | None = Query(None)):
        now = knowledge.utcnow()
        with svc.db() as conn:
            view = knowledge.MapView(conn, s, now)

            def cells(nid):
                return {m: {"mean": round(v[0], 3), "evidence": round(v[1], 2),
                            "own": [round(view.own(nid, m).alpha, 2), round(view.own(nid, m).beta, 2)]}
                        for m, v in view.agg.get(nid, {}).items()}

            if node_id is None:
                return {"node": None, "children": [{"id": r["id"], "name": r["name"], **view.summary(r["id"])}
                                                   for r in db.children(conn, None)]}
            node = db.get_node(conn, node_id)
            if node is None:
                raise HTTPException(404, "no such node")
            return {"node": {"id": node["id"], "name": node["name"], "description": node["description"],
                             "depth": node["depth"], "source": node["source"],
                             "path": [n["name"] for n in db.path(conn, node_id)],
                             "mastery": view.summary(node_id), "cells": cells(node_id)},
                    "children": [{"id": r["id"], "name": r["name"], **view.summary(r["id"])}
                                 for r in db.children(conn, node_id)]}

    @app.get("/api/next-steps")
    def next_steps_route(node_id: int):
        with svc.db() as conn:
            node = db.get_node(conn, node_id)
            if node is None:
                raise HTTPException(404, "no such node")
            vec = db.from_blob(node["embedding"])
        return {"next_steps": next_steps(node_id, vec, knowledge.utcnow())}

    @app.post("/api/replay")
    def replay():
        """Rebuild the whole map from `interactions` (e.g. after changing the weights in config)."""
        with svc.db() as conn:
            return {"events": knowledge.replay(conn, s)}

    # ------------------------------------------------------------ files

    @app.get("/audio/{name}")
    def audio_file(name: str):
        if not re.fullmatch(r"[0-9a-f]{8,64}\.(mp3|wav)", name):
            raise HTTPException(404)
        f = s.audio_dir / name
        if not f.exists():
            raise HTTPException(404)
        return FileResponse(f, media_type="audio/mpeg" if name.endswith(".mp3") else "audio/wav",
                            headers={"Cache-Control": "public, max-age=31536000, immutable"})

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def __getattr__(name: str):
    # `uvicorn app.main:app` builds the app on first access, so importing this module stays cheap.
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
