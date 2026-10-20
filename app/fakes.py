"""Offline stand-ins for Claude and ElevenLabs (MNEMO_FAKE=1, and the tests).

They answer quickly and deterministically, and they look at the prompt just enough for the
whole loop to behave sensibly: the fake recap reports the level it was asked for and leads with a
diagnosed gap, the fake judge flags proper nouns as "more specific", the quiz asks real
questions from `fake_quiz.json` and grades free text by its key ideas, and so on.
"""
from __future__ import annotations

import io
import json
import re
import wave
from pathlib import Path

from .tts import AudioCache, content_hash

QUIZ_BANK = {k: v for k, v in json.loads((Path(__file__).parent / "fake_quiz.json").read_text()).items()
             if not k.startswith("_")}
SHORT_KEYWORDS = {v["short"][0]: v["short"][2] for v in QUIZ_BANK.values()}

CPS = 15.0   # fake speech rate, characters per second
STARTERS = {"Remind", "Tell", "What", "Why", "How", "Explain", "The", "A", "An", "Please", "Who", "When"}


def topic_of(query: str) -> str:
    q = re.sub(r"^(please\s+)?(remind me (about|of)|tell me (about|of)|what (is|are)|explain)\s+", "", query.strip(),
               flags=re.I)
    return q.strip(" ?.!").lower() or "topic"


class FakeLLM:
    def __init__(self):
        self.calls: list[dict] = []
        self.asked: set[str] = set()       # quiz questions already given, so the bank isn't repeated

    # -------------------------------------------------------------- text

    def text(self, task, system, prompt, model, max_tokens=4000, effort="low"):
        self.calls.append({"task": task, "system": system, "prompt": prompt, "model": model})
        if task == "recap":
            return self._recap(prompt)
        if task == "script":
            plain = re.sub(r"\$\$?([^$]+)\$\$?", "a formula", prompt)
            plain = re.sub(r"[*_#>`-]", "", plain)
            return re.sub(r"\s+", " ", plain).strip()
        return "ok"

    def _recap(self, prompt: str) -> str:
        query = re.search(r"Query: (.*)", prompt).group(1)
        profile = json.loads(prompt.split("Knowledge profile:\n", 1)[1])
        mean = profile["node"]["mastery_by_mode"][profile["mode"]]["mean"]
        level = re.search(r"Pitch: (.*?):", prompt).group(1)
        kind = "An explanation" if "Intent: explain" in prompt else "A recap"
        lines = []
        for gap in profile["recent_gaps"][:1]:
            lines.append(f"**First, {gap['term']}.** This is the piece that was missing last time.")
        lines.append(f"*Level: {level}* (mastery {mean:.2f}). {kind} of **{topic_of(query)}**, "
                     f"in the {profile['mode']} mode, within {profile['topic']}.")
        lines.append("The central relation is $E = mc^2$, and the next layer is what happens beyond it. "
                     "Each idea builds on the one before. That is the frontier.")
        return "\n\n".join(lines)

    # -------------------------------------------------------------- json

    def json(self, task, system, prompt, schema, model, max_tokens=4000, effort="low"):
        self.calls.append({"task": task, "system": system, "prompt": prompt, "model": model})
        return getattr(self, "_" + task)(prompt)

    def _judge(self, prompt):
        query = re.search(r"Query: '(.*)'|Query: \"(.*)\"", prompt)
        query = next(g for g in query.groups() if g is not None)
        cands = re.findall(r"- id (\d+): (.*?) — ", prompt)
        q = query.lower()
        mode = ("explanatory" if re.search(r"\bwhy\b|\bhow does\b|\bexplain", q) else
                "practical" if re.search(r"\bhow (do|to)\b|recipe|cook", q) else
                "normative" if re.search(r"\bshould\b|\bgood\b|\bright\b", q) else "descriptive")
        names = " ".join(p.lower() for _, p in cands)
        proper = [w for w in re.findall(r"\b[A-Z][a-z]+", query)
                  if w not in STARTERS and w.lower() not in names]
        weights = {m: (0.6 if m == mode else 0.1) for m in
                   ("descriptive", "explanatory", "normative", "practical", "expressive")}
        return {"node_id": int(cands[0][0]), "confidence": 0.8, "mode": mode, "mode_weights": weights,
                "more_specific": bool(proper), "specific_topic": " ".join(proper)}

    def _refine(self, prompt):
        m = re.search(r"Query: '(.*)'|Query: \"(.*)\"", prompt)
        if m:
            query = next(g for g in m.groups() if g is not None)
            return {"children": [{"name": topic_of(query), "description": f"All about {topic_of(query)}."},
                                 {"name": "overview", "description": "The general picture."},
                                 {"name": "history", "description": "How it developed."}],
                    "match_index": 0, "still_more_specific": False}
        return {"children": [{"name": "foundations", "description": "The basic ideas."},
                             {"name": "key figures", "description": "The people who shaped it."},
                             {"name": "open questions", "description": "What is still debated."}],
                "match_index": -1, "still_more_specific": False}

    def _diagnose(self, prompt):
        note = re.search(r"Her note: (.*)", prompt)
        term = note.group(1).strip() if note and note.group(1).strip() not in ("", "(none)") else "partition function"
        return {"kind": "missing_prerequisite", "term": term, "explanation": f"The recap assumed {term}."}

    def _quiz_item(self, prompt):
        """A real question from fake_quiz.json for the deepest topic on the path that has one
        left in the requested format; the old generic stand-in otherwise."""
        path = re.search(r"Topic: (.*?) —", prompt).group(1).split(" › ")
        kind = "choice" if "multiple choice" in prompt else "short"
        for name in reversed(path):
            entry = QUIZ_BANK.get(name, {}).get(kind)
            if entry and entry[0] not in self.asked:
                self.asked.add(entry[0])
                if kind == "choice":
                    return {"question": entry[0], "options": entry[1:5], "correct_index": 0, "answer": entry[1]}
                return {"question": entry[0], "options": [], "correct_index": -1, "answer": entry[1]}
        topic = path[-1]
        if kind == "choice":
            return {"question": f"Which statement about {topic} is true?",
                    "options": ["The true one", "A false one", "Another false one", "A distractor"],
                    "correct_index": 0, "answer": "The true one"}
        return {"question": f"In one sentence, what is central to {topic}?", "options": [], "correct_index": -1,
                "answer": "The central idea."}

    def _grade(self, prompt):
        question = re.search(r"Question: (.*)", prompt).group(1).strip()
        their = prompt.rsplit("Their answer: ", 1)[1].strip().lower()
        keywords = SHORT_KEYWORDS.get(question, ["central"])
        hits = sum(any(k in their for k in group.split("|")) for group in keywords)
        return {"score": hits / len(keywords), "rationale": f"Fake grader: {hits} of {len(keywords)} key ideas."}


def silent_wav(seconds: float, rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(rate)
        w.writeframes(b"\x80" * int(seconds * rate))
    return buf.getvalue()


class FakeTTS:
    """Silent audio at CPS characters per second, with exact timings."""
    name = "fake"
    available = True

    def __init__(self, cache: AudioCache):
        self.cache = cache
        self.requests = 0

    def synthesize(self, text: str) -> dict:
        key = content_hash("fake", text)
        hit = self.cache.get(key)
        if hit:
            return {"url": f"/audio/{hit['file']}", "starts": hit["starts"], "cached": True}
        self.requests += 1
        starts = [round(i / CPS, 3) for i in range(len(text))]
        info = self.cache.put(key, silent_wav(len(text) / CPS + 0.3), "wav", starts, text)
        return {"url": f"/audio/{info['file']}", "starts": info["starts"], "cached": False}
