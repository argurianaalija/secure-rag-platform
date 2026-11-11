"""Shared test fixtures.

We override Settings to use a temporary upload folder and a small size limit,
so tests are fast, isolated and do not pollute the real `uploads/` folder.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        max_upload_size_mb=1,  # 1 MB: easy to exceed in tests
        upload_dir=str(tmp_path / "uploads"),
        environment="test",
        # Small chunks, so even a short text yields multiple blocks and we can
        # verify the overlap.
        chunk_size_chars=100,
        chunk_overlap_chars=20,
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
