"""Reusable semantic-search ingestion; no application startup on import."""

from .ingestion import generate_embedding, ingest, load_document, upsert_document

__all__ = ["load_document", "generate_embedding", "upsert_document", "ingest"]
