CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id          BIGSERIAL PRIMARY KEY,
    title       TEXT NOT NULL UNIQUE,
    content     TEXT NOT NULL,
    embedding   vector(384),    -- matches all-MiniLM-L6-v2 output dim
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_documents_metadata ON documents USING gin (metadata);
