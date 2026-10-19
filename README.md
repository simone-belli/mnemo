# Mnemo

*Remember what you know.*

Mnemo is a personal "external hard drive" for what I know. Ask it "remind me about Renaissance painting" and it gives a short recap pitched at my level. It reads the recap aloud while highlighting each word as it's spoken, and it suggests where to go next. Behind the scenes it keeps a calibrated map of my knowledge, and every interaction updates that map. See [SPEC.md](SPEC.md) for the full design.

It's a tool for one person, served locally: FastAPI, SQLite, Claude, ElevenLabs and one plain HTML page, with no build step.

## Run it

```
poetry install --without dev  # or plain `poetry install` to include the test tools
cp .env.example .env          # add ANTHROPIC_API_KEY and ELEVENLABS_API_KEY
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000. To use it from an iPhone on the same network, open `http://<your-mac>.local:8000`. The microphone needs HTTPS on iOS, so either use the text box or put the server behind something like `tailscale serve`.

- The first run seeds the knowledge tree from `app/seed_taxonomy.json` and embeds it locally. The first time, this downloads `BAAI/bge-small-en-v1.5`.
- Without an ElevenLabs key, the browser's own voice reads the recap, still with word highlighting.
- `MNEMO_FAKE=1 poetry run uvicorn app.main:app` runs with offline stand-ins for Claude, ElevenLabs and the embedder. Use it to try the interface without keys.
- Add `?debug=1` to the URL to open the debug drawer. It shows the tag distribution, the descent trace, the profile and prompt sent to the model, and the mastery deltas each action applied.

## Using it

- **Ask** by typing or with the mic.
- **Choose a mode:**
  - **Read** shows the formatted recap and generates no audio.
  - **Listen** plays the audio script with the current word highlighted.
  - **Read + Listen** does the same, with the formatted recap folded underneath.
- **Tap a word** in the script to jump there.
- **Give feedback** on every recap with *Deeper*, *Simpler*, *Knew that* or *Lost me*. *Lost me* asks for an optional note and then diagnoses what was missing.
  - Replaying a recap with ↻, or saying "repeat", counts as a mild "too hard" signal.
  - While a recap is on screen, the same words work as voice commands.
- **Next** chips are topics just past the edge of what you know. Tap one to recall it.
- **Recalibrate** (in the footer) runs the onboarding again: a self-rating per domain, then an adaptive quiz you can stop at any time.

## How it works

| File | What it does |
|---|---|
| `app/main.py` | Routes (SPEC §9) and the recall flow: tag → profile → recap (with next steps in parallel) → audio script → TTS |
| `app/config.py` | All settings and feedback weights, read from `.env` |
| `app/db.py` | SQLite schema (SPEC §8, plus `ref_id`/`deltas` on interactions and a `meta` table) and queries |
| `app/mastery.py` | Pure maths: Beta cells, decay, upward aggregation, child priors, frontier, quiz deltas |
| `app/knowledge.py` | The map in the database: effective mastery, events → deltas, profile, frontier, **replay** |
| `app/taxonomy.py` | Seeding, embeddings, lazy refinement (the LLM proposes children of a node) |
| `app/tagging.py` | Top-down descent, LLM judge (node, mode, "more specific?"), refinement, session smoothing |
| `app/recap.py` | Recap and audio-script prompts |
| `app/feedback.py` | The four buttons and the *Lost me* diagnosis |
| `app/quiz.py` | Self-rating, item selection by Beta variance × breadth, generation and grading |
| `app/tts.py` | ElevenLabs `with-timestamps`, cached in `data/audio/` by content hash |
| `app/embed.py` | Embedding interface: sentence-transformers, or a dependency-free hash embedder |
| `app/llm.py` | Thin Anthropic wrapper: text, schema-constrained JSON, retries |
| `app/fakes.py` | Offline Claude and ElevenLabs for `MNEMO_FAKE=1` and the tests |
| `static/` | `index.html`, `app.js`, `style.css`, `logo/` |

Every change to the map is an event in `interactions`. Self-ratings, quiz answers and feedback are all stored there, alongside the creation time of each lazily created node. `POST /api/replay` rebuilds `mastery` and `preferences` from those events using the current weights in `config.py`, so the weights can be retuned after the fact.

## Tests

```
poetry install
cd tests && npm install && cd ..      # local copies of marked and KaTeX for the browser test
poetry run pytest tests
```

- `test_mastery.py` covers the update rules, decay, aggregation, priors and frontier ranking.
- `test_flow.py` runs the acceptance criteria through the API, using fake Claude and ElevenLabs:
  - after two "knew that", the next recap is more advanced
  - after "lost me", the next recap leads with the prerequisite
  - a vague query tags shallow and a specific one tags deep
  - replaying the history rebuilds the map exactly
- `test_llm.py` checks the shape of the Anthropic requests.
- `test_ui.py` drives the page in Chromium at iPhone size:
  - onboarding and the quiz
  - recap rendering with KaTeX
  - word highlighting moving with the audio
  - feedback, modes and next-step chips

## Notes and open questions

- Logos live in `static/logo/`: `mnemo-mark.svg` (the full piazza scene), `mnemo-icon.svg` (inlined in the header of `index.html`), `favicon.svg` and `apple-touch-icon.png` (rendered from the icon for the iPhone home screen). The μ in each SVG is the EB Garamond glyph converted to a path, so it looks right even where the font can't load.
- Tagging thresholds (`TAG_TEMPERATURE`, `TAG_THRESHOLD`) need tuning against the real embedder. Use the debug drawer.
- `SAY_WHAT_I_KNOW` switches recaps between "You know X, so…" openings and just pitching at the right level.
- The script for the ear is written by `SCRIPT_MODEL`, which defaults to the fast model to keep recall under 20 seconds. Set it to the recap model for better scripts.
