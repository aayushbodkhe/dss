"""
Build or rebuild Chroma semantic index from data/guidelines.json.

Usage:
  python -m src.build_chroma_index
  python -m src.build_chroma_index --rebuild
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chroma semantic index for clinical guidelines")
    parser.add_argument("--rebuild", action="store_true", help="Delete and recreate the collection")
    parser.add_argument("--deployment", help="Azure OpenAI embedding deployment name override")
    parser.add_argument("--endpoint", help="Azure OpenAI endpoint override")
    parser.add_argument("--api-version", help="Azure OpenAI API version override")
    args = parser.parse_args()

    load_dotenv()

    try:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain_openai import AzureOpenAIEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies. Install requirements including langchain-chroma and chromadb."
        ) from exc

    root_dir = Path(__file__).resolve().parent.parent
    guidelines_path = root_dir / "data" / "guidelines.json"
    if not guidelines_path.exists():
        raise FileNotFoundError(f"Guidelines file not found: {guidelines_path}")

    with open(guidelines_path, "r", encoding="utf-8") as f:
        guidelines = json.load(f)

    endpoint = args.endpoint or os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
    deployment = args.deployment or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
    api_version = args.api_version or os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-02-01-preview")

    if not endpoint:
        endpoint = require_env("AZURE_OPENAI_ENDPOINT")
    if not api_key:
        api_key = require_env("AZURE_OPENAI_API_KEY")
    if not deployment:
        deployment = require_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    persist_dir = Path(os.getenv("CHROMA_PERSIST_DIR", str(root_dir / "configs" / "chroma")))
    collection_name = os.getenv("CHROMA_COLLECTION", "clinical_guidelines")
    persist_dir.mkdir(parents=True, exist_ok=True)

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=endpoint,
        api_key=api_key,
        azure_deployment=deployment,
        openai_api_version=api_version,
    )

    # Fail fast with a targeted diagnostic before indexing the full dataset.
    try:
        embeddings.embed_query("chroma health check")
    except Exception as exc:
        message = str(exc)
        if "404" in message or "Resource not found" in message:
            print("ERROR: Azure embedding resource not found (HTTP 404).", file=sys.stderr)
            print(f"  endpoint   : {endpoint}", file=sys.stderr)
            print(f"  deployment : {deployment}", file=sys.stderr)
            print(f"  api_version: {api_version}", file=sys.stderr)
            print("", file=sys.stderr)
            print("Likely causes:", file=sys.stderr)
            print("  1) Deployment name is incorrect (must be Azure deployment name, not model name).", file=sys.stderr)
            print("  2) Endpoint points to a different Azure OpenAI resource.", file=sys.stderr)
            print("  3) API version is not supported for your deployment.", file=sys.stderr)
            print("", file=sys.stderr)
            print("Try:", file=sys.stderr)
            print("  python -m src.build_chroma_index --deployment <your_embedding_deployment>", file=sys.stderr)
            print("", file=sys.stderr)
            sys.exit(2)
        raise

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    if args.rebuild:
        vectorstore.delete_collection()
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )

    docs = [
        Document(
            page_content=gl.get("body", ""),
            metadata={
                "id": gl.get("id", ""),
                "title": gl.get("title", ""),
                "category": gl.get("category", ""),
                "keywords": ", ".join(gl.get("keywords", [])),
            },
        )
        for gl in guidelines
    ]
    ids = [gl.get("id", f"GL_{idx}") for idx, gl in enumerate(guidelines)]

    # Replace existing ids to keep index deterministic and idempotent.
    vectorstore.delete(ids=ids)
    vectorstore.add_documents(docs, ids=ids)

    print(f"Indexed {len(docs)} guidelines into collection '{collection_name}' at {persist_dir}")


if __name__ == "__main__":
    main()
