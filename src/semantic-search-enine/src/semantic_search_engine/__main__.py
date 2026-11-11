"""Standalone entry point using the same four-function ingestion library."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

import asyncpg

from . import ingest, load_document


async def _run(documents: list[dict], database_url: str, model_cache: str | None) -> dict:
    connection = await asyncpg.connect(database_url)
    try:
        return await ingest(documents, connection, cache_folder=model_cache)
    finally:
        await connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest UTF-8 .txt and .md files recursively, one vector per file; no chunking.",
        epilog=("Connection: DATABASE_URL takes precedence; otherwise use POSTGRES_PASSWORD "
                "and Compose's POSTGRES_USER, POSTGRES_DB, POSTGRES_PORT defaults on 127.0.0.1."),
    )
    parser.add_argument("directory", type=Path, help="Source directory; relative file paths are unique titles.")
    parser.add_argument("--model-cache", help="Optional local sentence-transformers model cache directory.")
    args = parser.parse_args(argv)
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        password = os.environ.get("POSTGRES_PASSWORD", "")
        if not password:
            parser.error("Set POSTGRES_PASSWORD for local Compose, or provide DATABASE_URL explicitly.")
        user = os.environ.get("POSTGRES_USER") or "semantic_search"
        database = os.environ.get("POSTGRES_DB") or "semantic_search"
        port = os.environ.get("POSTGRES_PORT") or "5432"
        database_url = (
            f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}"
            f"@127.0.0.1:{port}/{quote(database, safe='')}"
        )
    if not args.directory.is_dir():
        parser.error("directory must be an existing directory.")
    try:
        paths = sorted(
            path for path in args.directory.rglob("*")
            if path.is_file() and path.suffix.lower() in {".txt", ".md"}
        )
        if not paths:
            parser.error("directory contains no .txt or .md documents.")
        documents = [load_document(path, root=args.directory) for path in paths]
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    try:
        stats = asyncio.run(_run(documents, database_url, args.model_cache))
    except Exception:
        # Avoid printing connection URLs, credentials, or document content.
        print("Ingestion failed. Check the database, schema, and model availability.", file=sys.stderr)
        return 1
    print(json.dumps(stats, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
