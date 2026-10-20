# Mnemo: version zero spec

> *Remember what you know.*

Mnemo is a personal "external hard drive" for what I know. I ask it things like "remind me about Renaissance painting". Within 10 to 20 seconds it gives me a short recap pitched exactly at my level. It reads the recap aloud while highlighting the text, and it suggests where to go next. Behind the scenes it keeps a calibrated map of my knowledge, and every interaction updates that map.

This is a personal tool for one user (me), not a product. Keep the code simple, readable and hackable. Prefer clarity over cleverness.

---

## 1. Scope

### In v0
- A single-page web app, served locally.
- **Recall:** "remind me about X" returns a level-calibrated recap as text (Markdown + LaTeX) and as audio, with synced highlighting.
- **Knowledge map:** a hierarchical topic tree, crossed with five modes of knowing. Each cell holds a mastery estimate that has uncertainty and decays over time.
- **Tagging:** every query is mapped onto the tree, at the right depth and mode.
- **Feedback loop:** four buttons that update the map: *deeper*, *simpler*, *knew that*, *lost me*.
- **Onboarding quiz:** a short adaptive test that seeds the map.
- **Next steps:** two or three suggested topics just past my frontier.

### Out of v0 (later)
- Diagrams and plots in recaps.
- Visualising the knowledge map itself (sunburst or zoomable tree).
- Spaced-repetition nudges and notifications.
- Comparing knowledge maps between people.
- Multi-user support, accounts and deployment.

---

## 2. Stack

| Layer | Choice | Notes |
|---|---|---|
| Backend | Python 3.12 + FastAPI | A single process. |
| Storage | SQLite (via `sqlite3` or SQLModel) | One file: `data/mnemo.db`. |
| LLM | Claude via the Anthropic Python SDK | Recaps use `claude-sonnet-5-5`. Tagging, judging and grading use `claude-haiku-5-5`. Put model names in config. |
| Embeddings | `sentence-transformers`, running locally (e.g. `BAAI/bge-small-en-v1.5`) | Needs no extra API key. Hide it behind an interface so it can be swapped. |
| TTS | ElevenLabs, using the *with-timestamps* endpoint | Reuse the approach from my **Read Along** repo (`simone-belli/read-along`): character-level timings drive word highlighting. |
| Frontend | One `index.html` with vanilla JS and CSS | Use KaTeX for maths and `marked` for Markdown, both from a CDN. No build step. |
| Voice input | The browser Web Speech API | Fall back to the text box. |

Config goes in a `.env` file: `ANTHROPIC_API_KEY`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `RECAP_MODEL`, `FAST_MODEL`, `EMBED_MODEL`.

Suggested layout:

```
mnemo/
  app/
    main.py            # FastAPI routes
    config.py
    db.py              # schema + queries
    taxonomy.py        # tree, seeding, lazy expansion
    tagging.py         # query -> (node, mode) distribution
    mastery.py         # Beta updates, decay, aggregation, frontier
    recap.py           # prompt building + LLM calls
    quiz.py            # onboarding adaptive test
    tts.py             # ElevenLabs + timestamps
    llm.py             # thin Anthropic wrapper (retries, JSON mode)
    embed.py           # embedding interface
  static/
    index.html
    app.js
    style.css
    logo/              # Mnemo piazza logo SVGs (provided)
  data/                # sqlite db, cached audio (gitignored)
  tests/
  SPEC.md
  README.md
```

---

## 3. The knowledge model

### 3.1 Two axes
- **Domain axis (the tree):** what the knowledge is about. The six roots are:
  1. `formal`: logic, mathematics, abstract structures
  2. `nature`: physics through biology
  3. `mind`: perception, psychology, consciousness
  4. `society`: history, politics, economics, language
  5. `artefacts`: technology, art, pop culture, things humans make
  6. `ultimate`: metaphysics, religion, meaning
- **Mode axis (fixed, not recursive):** what kind of knowing.
  `descriptive` (what is or was) · `explanatory` (why) · `normative` (what's good or right) · `practical` (how to do it) · `expressive` (what it's like).

Only the domain axis recurses. Every node, at every depth, carries all five modes.

### 3.2 The tree
- Seed it with the 6 roots plus around 8 to 12 second-level fields each. Generate these once with the LLM and save them to `app/seed_taxonomy.json` so I can hand-edit it.
- **Lazy refinement:** only create children when they're needed. That happens when a query tags to a node with high confidence but is clearly more specific than the node. The LLM then proposes 3 to 8 children of that node, and the matching one is selected. This is adaptive mesh refinement for knowledge.
- Each node has a name, a one-line description, a parent, a depth and an embedding of `"{name}: {description}"`.
- A node is a **child** of A if it's subsumed by A (texts about B almost always involve A, but not vice versa). When the LLM proposes children, the prompt should state this rule explicitly.

### 3.3 Mastery: Beta with forgetting
For each (node, mode) cell, store `alpha`, `beta` and `last_seen`.
- The prior is `Beta(1, 1)`.
- The **mastery mean** is `m = α / (α + β)`.
- The **uncertainty** is the Beta variance. It's shown as a confidence indicator and used to pick quiz items.
- **Decay on read:** shrink the evidence back towards the prior with half-life `h` (default 120 days, configurable):
  `α_eff = 1 + (α − 1) · 2^(−Δt/h)`, and likewise for β.
- **Aggregation upwards:** a parent's effective mastery is the evidence-weighted mean of its own cells and its children's cells. Weight each by its evidence `α + β − 2`.
- **Node-level mastery** is the evidence-weighted mean across the five modes.
- **Correlation prior for new nodes:** a newly created child starts at its parent's effective mean, scaled to a weak prior of 2 pseudo-counts total, rather than at `Beta(1,1)`.

All of this lives in `mastery.py` as pure functions with unit tests.

### 3.4 Frontier
A node is on the **frontier** when its own mastery is low (< 0.4) or very uncertain, while its parent or siblings are well known (> 0.65). Next steps are the top 2 or 3 frontier nodes near the current query. Rank them by `similarity × (parent_mastery − own_mastery)`.

---

## 4. Tagging pipeline (`tagging.py`)

Input: the query text, plus the last few turns for context.
Output: a distribution over `(node_id, mode)` with confidences, plus the chosen primary cell.

1. **Embed** the query.
2. **Descend from the top down:** at each level, score the children by cosine similarity and softmax them (temperature configurable). Descend while the best child's probability is above a threshold (default 0.5) and better than staying put. Stop when confidence drops. This sets the depth automatically.
3. **Keep the distribution** at the stopping level, not just the argmax. A query can be a genuine mixture.
4. **LLM judge (fast model, JSON output):** given the query and the top 5 candidate nodes, the judge confirms or corrects the node. It also assigns the **mode** (embeddings are poor at this), and it flags whether the query is more specific than every candidate. In that case, trigger lazy refinement (§3.2).
5. **Smoothing:** within a session, blend with the previous turn's distribution (EMA, α = 0.6) so tags don't flicker.

---

## 5. Recall flow (`recap.py`)

1. `POST /api/recall {query}` → tag (§4).
2. **Build the profile context** for the primary node:
   - the node's own mastery per mode, and its uncertainty
   - the parent's and siblings' mastery (what I know around it)
   - its children (what's below it), with mastery
   - recent "lost me" events on this branch, with the diagnosed gaps
   - my depth preference for this branch (from *deeper* and *simpler* feedback)
3. **Generate the text recap** (recap model). Default length is 120 to 180 words. Use Markdown, and LaTeX where it helps (I'm a physicist, so formulas are welcome, not scary). Pitch it one step past my edge: compress what I already know into a line, then spend most of the words on the next layer.
4. **Generate the audio script** with a *separate* call. It's the same content written for the ear: short sentences, verbal signposting, formulas described in words, and no Markdown.
5. **TTS** on the audio script, using ElevenLabs with timestamps. Cache the audio by content hash in `data/audio/`.
6. Return `{text_md, audio_script, audio_url, timings, node_path, mode, next_steps[]}`.

**Sync highlighting when text and audio differ:** highlight the audio script in "listen" mode. In "read + listen" mode, show the audio script as the main text, with the formatted text recap collapsible below it. (Simple, and avoids alignment hell. Revisit later.)

### Draft system prompt for the recap

```
You are Mnemo, a personal memory companion for one person: Simone, a
theoretical physicist (PhD, statistical physics) turned quant researcher,
with broad interests in literature, philosophy, art and cooking.

Your job: remind her what she knows about a topic and push exactly one
step past her current edge. Be brief, dense and precise. No preamble,
no "great question", no summary at the end.

You will receive her knowledge profile for this topic as JSON:
mastery per mode (0-1) with uncertainty, the surrounding branch, recent
gaps, and her depth preference.

Rules:
- Where mastery is high: compress to one line, as a reminder.
- Where mastery is low: introduce gently, high-level first, name the
  key terms.
- Spend most words on the frontier: the next layer she doesn't yet have.
- If a recent gap was diagnosed (a missing prerequisite), cover it first.
- Formulas in LaTeX are welcome when they carry the idea.
- Target {length} words. Markdown allowed.
```

---

## 6. Feedback loop

The UI shows four buttons under every recap (voice commands work too): **Deeper**, **Simpler**, **Knew that**, **Lost me**. Asking for a repeat counts as an implicit signal.

| Signal | Update |
|---|---|
| Knew that | `α += 1.0` on the primary cell |
| Deeper | `α += 0.5`, and nudge this branch's depth preference up |
| Simpler | `β += 0.5`, and nudge the depth preference down |
| Lost me | Run a **diagnosis** call (fast model): was it a missed word, an unfamiliar term, or a missing prerequisite? Unfamiliar term → `β += 0.5` on the node for that term (tag it). Missing prerequisite → `β += 1.0` on the prerequisite node (tag it, create it if needed). Only lower the whole subtopic when it's a genuine gap. Log the diagnosis for the profile context. |
| Repeat request | `β += 0.25` on the primary cell |
| Simply asked about it | No mastery change. Update `last_seen` only. |

Weights live in config. Every event is stored in `interactions` so the map can be rebuilt from scratch, and the weights can be tuned later by replaying the history.

---

## 7. Onboarding quiz (`quiz.py`)

- Target **15 to 25 questions** in about 10 minutes, which can be stopped at any time.
- **Item selection:** choose the (node, mode) cell with the highest expected information gain. For v0, the highest Beta variance weighted by node breadth is fine. Start at the second level of the tree, and go one level deeper wherever answers are strong.
- **Item generation:** the fast model writes one question for the cell at a target difficulty. Mix multiple choice (4 options) with short free text. Difficulty is absolute, on a scale with concrete anchors in the prompt (0 is general knowledge, 1 a specialist).
- **Adaptive difficulty:** a staircase per root domain. It starts easy (0.25). A right answer moves it up 0.1, a wrong one or *I don't know* down 0.15, and partial credit in between, so it settles where about 60% of answers are right. A domain's first question starts at the overall level reached so far.
- **Grading:** multiple choice is graded exactly. Free text is graded by the fast model on a 0 to 1 scale with a one-line rationale. Update with `α += score`, `β += 1 − score`, scaled by item difficulty.
- **Correlation spread:** a strong answer gives the siblings a small boost (+0.2 α), because someone strong on Renaissance painting is probably decent on the Reformation.
- There's **no self-rating** screen, so the calibration isn't biased by self-assessment. (Earlier versions had one; their `rating` events still replay.)
- The quiz can be rerun later as a "recalibrate" option. Each run is a new round of up to 20 questions, and its difficulty carries on from the earlier answers.

---

## 8. Data model (SQLite)

```sql
nodes(id INTEGER PK, parent_id INTEGER NULL, name TEXT, description TEXT,
      depth INTEGER, embedding BLOB, created_at TEXT, source TEXT)  -- seed|lazy|quiz
mastery(node_id INTEGER, mode TEXT, alpha REAL, beta REAL, last_seen TEXT,
        PRIMARY KEY(node_id, mode))
interactions(id INTEGER PK, ts TEXT, kind TEXT,      -- recall|feedback|quiz
             query TEXT, node_id INTEGER, mode TEXT, tag_dist TEXT,  -- JSON
             response_md TEXT, audio_script TEXT, audio_path TEXT,
             feedback TEXT NULL, diagnosis TEXT NULL)               -- JSON
preferences(branch_node_id INTEGER PK, depth_pref REAL)  -- -1..+1
quiz_items(id INTEGER PK, node_id INTEGER, mode TEXT, difficulty REAL,
           question TEXT, options TEXT NULL, answer TEXT NULL,
           user_answer TEXT, score REAL, ts TEXT)
```

---

## 9. API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/recall` | `{query}` returns the recap payload (§5) |
| POST | `/api/feedback` | `{interaction_id, signal, note?}` |
| GET | `/api/quiz/next` | returns the next item |
| POST | `/api/quiz/answer` | `{item_id, answer}` returns the score and next item |
| GET | `/api/map?node_id=` | returns a node, its children, and mastery (for debugging) |
| GET | `/api/next-steps?node_id=` | returns frontier suggestions |
| GET | `/audio/{hash}.mp3` | cached audio |

---

## 10. UI

A single page, mobile-friendly first: I'll use it from my iPhone as much as from my Mac.

- **Header:** the Mnemo logo (the piazza mark, `static/logo/`) and the wordmark.
- **Query bar:** a text field with a mic button (Web Speech API). The placeholder is *"Remind me about…"*.
- **Mode toggle:** Read · Listen · Read + Listen. Remember the last choice in localStorage.
- **Recap card:**
  - a breadcrumb showing the tagged path (e.g. *artefacts › art › painting › Renaissance*) with the mode as a small chip
  - the rendered Markdown and KaTeX
  - an audio player with synced word highlighting (port this from Read Along)
- **Feedback row:** Deeper · Simpler · Knew that · Lost me. Each button is at least 44 px tall and acknowledged with a subtle state change.
- **Next steps:** two or three chips under the card. Tapping one runs a recall on it.
- **First run:** the quiz flow (questions with a progress bar, an "I don't know" link and a "stop and start using Mnemo" link).
- **A debug drawer** (hidden behind `?debug=1`): the tag distribution, the profile JSON sent to the model, and the mastery deltas applied.

### Brand
| Token | Value | Use |
|---|---|---|
| `--ink` | `#1E3B34` | text, deep green, dusk sky |
| `--paper` | `#F2ECE0` | page background |
| `--stone` | `#E8D9B5` | cards, buildings |
| `--ochre` | `#D9A84E` | the piazza, highlights |
| `--shadow` | `#8A6526` | long shadows, borders |
| `--accent` | `#C2502E` | the red pennant: primary action, current highlighted word |

- **Fonts** (Google Fonts): *Fraunces italic* for the wordmark and headings. *EB Garamond* for body text and the μ.
- **Mood:** de Chirico. Quiet, metaphysical, warm. Use flat colour, no gradients and no emoji. Highlighting in audio should look like a warm lamp, not a neon marker: an ochre background and an ink text colour.
- The logo files are provided in `static/logo/`. The μ is set in EB Garamond, so make sure the font is loaded wherever the SVG is inlined, or convert the glyph to a path.

---

## 11. Milestones

Build them in this order, and each one should be usable on its own:

1. **Skeleton:** FastAPI + `index.html`. Recall calls Claude with no profile and renders the Markdown + KaTeX.
2. **Taxonomy + tagging:** the seed tree, embeddings, top-down descent, the LLM judge, the breadcrumb in the UI, and the debug drawer.
3. **Mastery + feedback:** the Beta cells, decay, aggregation, the four buttons, the profile context in the prompt, and lost-me diagnosis.
4. **Audio:** the ear-written script, ElevenLabs timestamps, caching, the three modes, and synced highlighting.
5. **Onboarding quiz:** adaptive items and grading.
6. **Next steps + lazy refinement:** frontier suggestions and on-demand child creation.

---

## 12. Acceptance criteria

- I can ask "remind me about Renaissance painting" on my phone and hear and read a recap in under 20 seconds (cold) or under 5 seconds (cached audio).
- After I say "knew that" twice on a topic, the next recap on it is visibly more advanced: the debug drawer shows the mastery change, and the prompt reflects it.
- "Lost me" on a recap that relies on an unfamiliar prerequisite logs a diagnosis that names that prerequisite, and the next recap covers it first.
- A vague query ("tell me about art") tags at a shallow depth. A specific one ("Brunelleschi's perspective experiment") tags deep, creating nodes if needed.
- `mastery.py` has unit tests for the update rules, decay, aggregation and the frontier ranking.
- The whole map can be rebuilt by replaying `interactions`.

---

## 13. Open questions (decide while building; don't block on them)

- What should the tagging thresholds and softmax temperature be? Tune them using the debug drawer.
- Should a recap explicitly say *what* it thinks I know ("you know X, so...")? Make it a toggle and see which I prefer.
- Should the decay half-life differ by domain, since maths may persist longer than pop culture?
- What's the best way to align highlighting when the text and audio versions differ?
