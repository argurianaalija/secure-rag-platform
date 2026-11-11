"""Create the database schema (pgvector extension + table + index).

Usage (after `docker compose up -d`):
    python -m scripts.init_db
"""

from app.config import get_settings
from app import db


def main() -> None:
    settings = get_settings()
    db.init_schema(settings)
    print(f"Schema ready on {settings.database_url}")


if __name__ == "__main__":
    main()
