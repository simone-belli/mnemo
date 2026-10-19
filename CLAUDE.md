# Mnemo

A personal memory companion for one user (Simone Belli). You ask "remind me about X" and get back a short recap pitched at your level, as text and as audio with synced word highlighting. A calibrated knowledge map is updated by every interaction. The design is in `SPEC.md`; `README.md` has the file map.

## Ground rules

- Minor changes (small fixes, tweaks, docs) can be pushed straight to `main`. Major changes (new features, schema or API changes, anything risky) go on a branch and land through a pull request so the owner can try them first.
- Keep it simple, readable and hackable: one FastAPI process, SQLite, one plain `static/index.html` + `app.js` + `style.css`. No build step and no framework. CDN libraries are pinned to exact versions.
- `app/mastery.py` stays pure (no database or I/O) and unit-tested.
- Every change to the map must come from an event in `interactions` and go through `knowledge.deltas_for()`, so `knowledge.replay()` can rebuild it. If you add a new kind of event, add it to `deltas_for()` and to `test_map_rebuilds_from_interactions`.
- Model names, weights and thresholds live in `app/config.py` / `.env`, never inline.
- Never commit `.env`, API keys or `data/`.
- Run `python3 -m pytest tests` before every push. The tests run offline with the fakes in `app/fakes.py`, and the browser test needs `cd tests && npm install`.

## Environment notes

- The cloud workspace can reach PyPI and npm, but not the jsDelivr CDN, Hugging Face model downloads or (without a key) the Claude and ElevenLabs APIs. Use the fakes and the hash embedder (`EMBED_MODEL=hash`), and ask the owner to try real calls when prompts or API contracts change.
- The ElevenLabs approach (character timings → word highlighting, cached audio) is ported from `simone-belli/read-along`.
