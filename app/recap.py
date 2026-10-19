"""Recaps (SPEC §5): the profile goes into the prompt, the recap model writes the text, and a
second call rewrites it for the ear."""
from __future__ import annotations

import json

from .config import Settings

RECAP_SYSTEM = """You are Mnemo, a personal memory companion for one person: Simone, a
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
- Target {length} words. Markdown allowed."""

MODE_HINTS = {
    "descriptive": "She is asking what is or was: facts, events, people, dates, structure.",
    "explanatory": "She is asking why or how it works: mechanisms, causes, theory.",
    "normative": "She is asking what is good or right: arguments, judgements, positions and their strongest objections.",
    "practical": "She is asking how to do it: method, steps, technique, pitfalls.",
    "expressive": "She is asking what it is like: the experience, texture and feel of it.",
}

SCRIPT_SYSTEM = """You turn a short written recap into a script that will be read aloud by a
text-to-speech voice. Same content, written for the ear:
- short sentences, one idea each; verbal signposting ("First...", "The key point is...");
- describe formulas in words ("the energy is the sum over pairs of J times s i times s j"),
  never LaTeX, symbols or code;
- no Markdown, no bullet points, no headings, no parentheses, no emoji;
- keep it about as long as the recap. Output only the script."""


def depth_words(depth_pref: float) -> str:
    if depth_pref > 0.3:
        return "She has been asking to go deeper on this branch: be more technical and advanced."
    if depth_pref < -0.3:
        return "She has been asking for simpler versions on this branch: be gentler and more concrete."
    return ""


def recap_prompt(settings: Settings, query: str, profile: dict) -> str:
    parts = [f"Query: {query}", f"Mode: {profile['mode']}. {MODE_HINTS.get(profile['mode'], '')}"]
    if hint := depth_words(profile.get("depth_preference", 0)):
        parts.append(hint)
    if settings.say_what_i_know:
        parts.append("Open with one short clause naming what she already knows (\"You know X, so...\").")
    else:
        parts.append("Don't tell her what she knows; just pitch the recap at that level.")
    parts.append("Knowledge profile:\n" + json.dumps(profile, ensure_ascii=False, indent=1))
    return "\n\n".join(parts)


def write_recap(llm, settings: Settings, query: str, profile: dict) -> tuple[str, str]:
    """-> (system prompt, Markdown recap)."""
    system = RECAP_SYSTEM.replace("{length}", f"{settings.recap_words - 30} to {settings.recap_words + 30}")
    return system, llm.text("recap", system, recap_prompt(settings, query, profile), settings.recap_model,
                            max_tokens=4000, effort="low")


def write_script(llm, settings: Settings, text_md: str) -> str:
    return llm.text("script", SCRIPT_SYSTEM, text_md, settings.script_model, max_tokens=3000, effort="low")
