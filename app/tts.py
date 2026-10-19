"""Speech with per-character timings (SPEC §5.5), the same approach as Read Along.

ElevenLabs' `with-timestamps` endpoint returns the audio and the start time of every character;
the page uses those to highlight the word being spoken. Audio and timings are cached in
`data/audio/` by a hash of model, voice and text, so replays cost no credits.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import httpx

EL_API = "https://api.elevenlabs.io/v1"


class TTSError(RuntimeError):
    pass


def content_hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]


class AudioCache:
    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict | None:
        meta = self.dir / f"{key}.json"
        if not meta.exists():
            return None
        info = json.loads(meta.read_text())
        return info if (self.dir / info["file"]).exists() else None

    def put(self, key: str, audio: bytes, ext: str, starts: list[float], text: str) -> dict:
        info = {"file": f"{key}.{ext}", "starts": starts, "text": text}
        (self.dir / info["file"]).write_bytes(audio)
        (self.dir / f"{key}.json").write_text(json.dumps(info))
        return info


class ElevenLabsTTS:
    name = "elevenlabs"

    def __init__(self, api_key: str, voice_id: str, model: str, cache: AudioCache):
        self.key, self.voice, self.model, self.cache = api_key, voice_id, model, cache

    @property
    def available(self) -> bool:
        return bool(self.key)

    def synthesize(self, text: str) -> dict:
        """-> {"url": "/audio/<hash>.mp3", "starts": [seconds per character], "cached": bool}"""
        key = content_hash(self.model, self.voice, text)
        hit = self.cache.get(key)
        if hit:
            return {"url": f"/audio/{hit['file']}", "starts": hit["starts"], "cached": True}
        try:
            r = httpx.post(
                f"{EL_API}/text-to-speech/{self.voice}/with-timestamps",
                params={"output_format": "mp3_44100_128"},
                headers={"xi-api-key": self.key},
                json={"text": text, "model_id": self.model},
                timeout=60,
            )
        except httpx.HTTPError as e:
            raise TTSError(f"cannot reach ElevenLabs: {e}") from e
        if r.status_code != 200:
            raise TTSError(f"ElevenLabs {r.status_code}: {r.text[:300]}")
        body = r.json()
        info = self.cache.put(key, base64.b64decode(body["audio_base64"]), "mp3",
                              align(text, body.get("alignment") or {}), text)
        return {"url": f"/audio/{info['file']}", "starts": info["starts"], "cached": False}


def align(text: str, alignment: dict) -> list[float]:
    """One start time per character of `text`, from ElevenLabs' alignment.

    The alignment's characters normally match the text one to one; if they don't, times are
    carried over by walking both strings and skipping characters that don't match."""
    chars = alignment.get("characters") or []
    starts = alignment.get("character_start_times_seconds") or []
    if "".join(chars) == text:
        return [round(float(s), 3) for s in starts]
    out, j, last = [], 0, 0.0
    for ch in text:
        k = j
        while k < len(chars) and chars[k] != ch and k - j < 8:
            k += 1
        if k < len(chars) and chars[k] == ch:
            last, j = float(starts[k]), k + 1
        out.append(round(last, 3))
    return out
