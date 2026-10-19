"""Settings, read once from the environment (and a `.env` file at the repo root).

Everything tunable lives here: model names, tagging thresholds, feedback weights,
the decay half-life. Change a value in `.env` rather than in code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODES = ("descriptive", "explanatory", "normative", "practical", "expressive")


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _b(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Weights:
    """Mastery updates for each feedback signal (SPEC §6). Replaying uses these too."""
    knew_that: float = 1.0      # alpha
    deeper: float = 0.5         # alpha
    simpler: float = 0.5        # beta
    lost_term: float = 0.5      # beta, on the unfamiliar term's node
    lost_prereq: float = 1.0    # beta, on the missing prerequisite's node
    lost_gap: float = 0.5       # beta, on the recap's own cell, only for a genuine gap
    repeat: float = 0.25        # beta
    depth_nudge: float = 0.25   # deeper/simpler move the branch's depth preference by this
    sibling_spread: float = 0.2  # alpha for siblings after a strong quiz answer
    rating_prior: float = 2.0   # pseudo-counts for a self-rating


@dataclass
class Settings:
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    elevenlabs_api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    elevenlabs_voice_id: str = field(default_factory=lambda: os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb"))
    elevenlabs_model: str = field(default_factory=lambda: os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5"))

    recap_model: str = field(default_factory=lambda: os.getenv("RECAP_MODEL", "claude-sonnet-5-5"))
    fast_model: str = field(default_factory=lambda: os.getenv("FAST_MODEL", "claude-haiku-5-5"))
    # The ear-written script is a rewrite of the recap, so the fast model is enough and keeps recall quick.
    script_model: str = field(default_factory=lambda: os.getenv("SCRIPT_MODEL", os.getenv("FAST_MODEL", "claude-haiku-5-5")))
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))

    # MNEMO_FAKE=1 swaps Claude, ElevenLabs and the embedder for offline stand-ins (see app/fakes.py).
    fake: bool = field(default_factory=lambda: _b("MNEMO_FAKE", False))

    data_dir: Path = field(default_factory=lambda: Path(os.getenv("MNEMO_DATA", ROOT / "data")))

    # Tagging (SPEC §4); tune with the debug drawer.
    tag_temperature: float = field(default_factory=lambda: _f("TAG_TEMPERATURE", 0.05))
    tag_threshold: float = field(default_factory=lambda: _f("TAG_THRESHOLD", 0.5))
    tag_ema: float = field(default_factory=lambda: _f("TAG_EMA", 0.6))
    max_refine_steps: int = field(default_factory=lambda: int(_f("MAX_REFINE_STEPS", 3)))

    # Mastery (SPEC §3.3–3.4)
    half_life_days: float = field(default_factory=lambda: _f("HALF_LIFE_DAYS", 120))
    frontier_low: float = 0.4
    frontier_high: float = 0.65

    # Recap (SPEC §5)
    recap_words: int = field(default_factory=lambda: int(_f("RECAP_WORDS", 150)))
    say_what_i_know: bool = field(default_factory=lambda: _b("SAY_WHAT_I_KNOW", False))

    # Quiz (SPEC §7)
    quiz_target: int = field(default_factory=lambda: int(_f("QUIZ_TARGET", 20)))

    weights: Weights = field(default_factory=Weights)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "mnemo.db"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"
