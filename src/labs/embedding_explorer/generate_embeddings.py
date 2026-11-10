"""Genera embeddings del corpus con OpenAI y después con SentenceTransformer.

Dependencias: numpy, openai, sentence-transformers y python-dotenv.
Desde la raíz, con el entorno virtual activado:
    python src/labs/embedding_explorer/generate_embeddings.py

Completa OPENAI_API_KEY en el .env junto a este archivo, o en el entorno.
El .env no reemplaza variables ya definidas en el entorno. OpenAI envía los
textos a su API y genera un costo según uso (text-embedding-3-small).
SentenceTransformer descarga all-MiniLM-L6-v2 la primera vez y luego lo
ejecuta localmente; devuelve vectores normalizados. Solo se imprimen formas.
"""

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from corpus import TEXTS


def generate_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (embeddings_openai, embeddings_locales), una fila por texto."""
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "Define OPENAI_API_KEY en el archivo .env o en el entorno antes de generar embeddings."
        )

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model="text-embedding-3-small", input=TEXTS)
    openai_embeddings = np.array([item.embedding for item in response.data])

    model = SentenceTransformer("all-MiniLM-L6-v2")
    local_embeddings = np.asarray(model.encode(TEXTS, normalize_embeddings=True))
    return openai_embeddings, local_embeddings


def main() -> None:
    """Muestra únicamente las formas etiquetadas de ambos resultados."""
    openai_embeddings, local_embeddings = generate_embeddings()
    print(f"OpenAI: {openai_embeddings.shape}")
    print(f"SentenceTransformer: {local_embeddings.shape}")


if __name__ == "__main__":
    main()
