# Mnemo

*Recall what you know, understand what's next.*

Mnemo is a personal "external hard drive" for what I know, and a tutor for what's next. Type a topic, say "Renaissance painting", and press **Recall** for a short recap pitched at my level, or **Explain** for an explanation one level simpler that builds the idea up. It reads the answer aloud while highlighting each word as it's spoken, and it suggests where to go next. Behind the scenes it keeps a calibrated map of my knowledge, and every interaction updates that map. See [SPEC.md](SPEC.md) for the full design.

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
- `MNEMO_FAKE=1 poetry run uvicorn app.main:app` runs with offline stand-ins for Claude, ElevenLabs and the embedder. Use it to try the interface without keys. Its quiz asks real questions from `app/fake_quiz.json` (one of each kind per second-level topic), but recaps stay placeholders. Point it at a separate data folder (`MNEMO_DATA=data-fake`) so its made-up nodes stay out of your real map.
- Add `?debug=1` to the URL to open the debug drawer. It shows the tag distribution, the descent trace, the profile and prompt sent to the model, and the mastery deltas each action applied.

## Using it

- **Ask** by typing or with the mic, then choose:
  - **Recall** (or Enter): a dense reminder at your level that pushes one step past it.
  - **Explain**: an explanation one level simpler, with terms defined and an example. Asking for one counts as a mild "not fully known" signal on the map. Spoken queries that start with "explain" or "what is" explain.
  - The card shows the level it was pitched at, e.g. *explain · introductory*. The levels run first contact · introductory · intermediate · advanced · expert.
- **Choose a mode:**
  - **Read** shows the formatted recap and generates no audio.
  - **Listen** plays the audio script with the current word highlighted.
  - **Read + Listen** does the same, with the formatted recap folded underneath.
- **Tap a word** in the script to jump there.
- **Rate how hard it was** on every recap or explanation, from 1 (*knew it*, way too easy) through 3 (*spot on*) to 5 (*lost me*). Ratings update the map and move how deep the next recap on that branch goes. Only a 5 asks what was wrong (optional) and then diagnoses what was missing.
  - Replaying a recap with ↻, or saying "repeat", counts as a mild "too hard" signal.
  - While a recap is on screen, you can also rate by voice: "three", "rate it 4", "spot on", "too easy", "lost me".
- **Next** chips are topics just past the edge of what you know. Tap one to have it explained.
- **The quiz** runs on first use, and again whenever you press **Recalibrate** in the footer. It starts easy and adapts to your answers in each domain, and you can stop it at any time. There is no self-rating, to keep self-assessment bias out of the map.
  - *I don’t know* under a question shows the answer and counts as a wrong answer, which is better than guessing.
- **Theme** (in the footer) switches between *auto* (follows the system's light or dark setting), *light* and *dark*.

## How it works

| File | What it does |
|---|---|
| `app/main.py` | Routes (SPEC §9) and the recall/explain flow: tag → profile → recap or explanation (with next steps in parallel) → audio script → TTS |
| `app/config.py` | All settings and feedback weights, read from `.env` |
| `app/db.py` | SQLite schema (SPEC §8, plus `ref_id`/`deltas` on interactions and a `meta` table) and queries |
| `app/mastery.py` | Pure maths: Beta cells, decay, upward aggregation, child priors, frontier, quiz deltas, the difficulty staircase, levels |
| `app/knowledge.py` | The map in the database: effective mastery, events → deltas, profile, frontier, **replay** |
| `app/taxonomy.py` | Seeding, embeddings, lazy refinement (the LLM proposes children of a node) |
| `app/tagging.py` | Top-down descent, LLM judge (node, mode, "more specific?"), refinement, session smoothing |
| `app/recap.py` | Recall and explain prompts, the target level, and the audio-script prompt |
| `app/feedback.py` | The 1–5 difficulty rating and the *lost me* diagnosis |
| `app/quiz.py` | Item selection by Beta variance × breadth, adaptive difficulty, generation and grading |
| `app/tts.py` | ElevenLabs `with-timestamps`, cached in `data/audio/` by content hash |
| `app/embed.py` | Embedding interface: sentence-transformers, or a dependency-free hash embedder |
| `app/llm.py` | Thin Anthropic wrapper: text, schema-constrained JSON, retries |
| `app/fakes.py` | Offline Claude and ElevenLabs for `MNEMO_FAKE=1` and the tests; `fake_quiz.json` is its question bank |
| `static/` | `index.html`, `app.js`, `style.css`, `logo/` |

Every change to the map is an event in `interactions`. Recalls, explanations, quiz answers and feedback are all stored there, alongside the creation time of each lazily created node. `POST /api/replay` rebuilds `mastery` and `preferences` from those events using the current weights in `config.py`, so the weights can be retuned after the fact.

## Tests

```
poetry install
cd tests && npm install && cd ..      # local copies of marked and KaTeX for the browser test
poetry run pytest tests
```

- `test_mastery.py` covers the update rules, decay, aggregation, priors, frontier ranking, the difficulty staircase and the levels.
- `test_flow.py` runs the acceptance criteria through the API, using fake Claude and ElevenLabs:
  - after two "too easy" ratings, the next recap is more advanced
  - ratings move mastery and the depth preference
  - an explanation is one level simpler than a recall of the same topic
  - after a 5 ("lost me"), the next recap leads with the prerequisite
  - a vague query tags shallow and a specific one tags deep
  - replaying the history rebuilds the map exactly
- `test_llm.py` checks the shape of the Anthropic requests.
- `test_ui.py` drives the page in Chromium at iPhone size:
  - onboarding and the quiz
  - recap rendering with KaTeX
  - word highlighting moving with the audio
  - the 1–5 scale (only a 5 asks what was wrong), modes and next-step chips
  - Explain beside Recall, and chips that explain
  - the theme switch

## Notes and open questions

- Logos live in `static/logo/`: `mnemo-mark.svg` (the full piazza scene), `mnemo-icon.svg` (inlined in the header of `index.html`), `favicon.svg` and `apple-touch-icon.png` (rendered from the icon for the iPhone home screen). The μ in each SVG is the EB Garamond glyph converted to a path, so it looks right even where the font can't load.
- Tagging thresholds (`TAG_TEMPERATURE`, `TAG_THRESHOLD`) need tuning against the real embedder. Use the debug drawer.
- `SAY_WHAT_I_KNOW` switches recaps between "You know X, so…" openings and just pitching at the right level.
- The script for the ear is written by `SCRIPT_MODEL`, which defaults to the fast model to keep recall under 20 seconds. Set it to the recap model for better scripts.
