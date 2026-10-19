import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.main import create_app, services  # noqa: E402


@pytest.fixture
def svc(tmp_path):
    """Fake Claude, fake ElevenLabs and the hash embedder, on a fresh database."""
    return services(Settings(fake=True, data_dir=tmp_path))


@pytest.fixture
def client(svc):
    return TestClient(create_app(svc))
