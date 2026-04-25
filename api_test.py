"""
API smoke test for Azure OpenAI chat and embeddings.

Usage examples:
  python -m src.api_test
  python -m src.api_test --chat-only
  python -m src.api_test --embedding-only
  python -m src.api_test --chat-deployment gpt-4o --embedding-deployment text-embedding-3-small
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import AzureOpenAI


@dataclass
class ApiConfig:
    endpoint: str
    chat_api_key: str
    chat_deployment: str
    chat_api_version: str
    embedding_api_key: str
    embedding_deployment: str
    embedding_api_version: str


def _get_env(*names: str, required: bool = False, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    if required:
        raise RuntimeError(f"Missing required environment variable. Tried: {', '.join(names)}")
    return default


def load_config(args: argparse.Namespace) -> ApiConfig:
    load_dotenv()

    endpoint = args.endpoint or _get_env("AZURE_OPENAI_ENDPOINT", required=True)

    chat_api_key = args.chat_api_key or _get_env(
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_KEY",
        required=True,
    )
    chat_deployment = args.chat_deployment or _get_env("AZURE_OPENAI_DEPLOYMENT", required=True)
    chat_api_version = args.chat_api_version or _get_env(
        "AZURE_OPENAI_API_VERSION",
        default="2024-12-01-preview",
    )

    embedding_api_key = args.embedding_api_key or _get_env(
        "AZURE_OPENAI_EMBEDDING_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_KEY",
        required=True,
    )
    embedding_deployment = args.embedding_deployment or _get_env(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        "AZURE_OPENAI_EMBEDDING_MODEL",
        required=True,
    )
    embedding_api_version = args.embedding_api_version or _get_env(
        "AZURE_OPENAI_EMBEDDING_API_VERSION",
        default="2024-02-01-preview",
    )

    return ApiConfig(
        endpoint=endpoint,
        chat_api_key=chat_api_key,
        chat_deployment=chat_deployment,
        chat_api_version=chat_api_version,
        embedding_api_key=embedding_api_key,
        embedding_deployment=embedding_deployment,
        embedding_api_version=embedding_api_version,
    )


def _print_result(name: str, ok: bool, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")


def test_chat(config: ApiConfig) -> bool:
    try:
        client = AzureOpenAI(
            api_key=config.chat_api_key,
            api_version=config.chat_api_version,
            azure_endpoint=config.endpoint,
        )
        response = client.chat.completions.create(
            model=config.chat_deployment,
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Reply with exactly: CHAT_OK"},
            ],
            max_tokens=10,
            temperature=0,
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            _print_result("Chat API", False, "No content returned")
            return False
        _print_result("Chat API", True, f"deployment={config.chat_deployment}, response='{content}'")
        return True
    except Exception as exc:
        _print_result("Chat API", False, str(exc))
        return False


def test_embeddings(config: ApiConfig) -> bool:
    try:
        client = AzureOpenAI(
            api_key=config.embedding_api_key,
            api_version=config.embedding_api_version,
            azure_endpoint=config.endpoint,
        )
        response = client.embeddings.create(
            model=config.embedding_deployment,
            input="clinical decision support embedding smoke test",
        )
        if not response.data or not response.data[0].embedding:
            _print_result("Embeddings API", False, "Empty embedding vector")
            return False
        dim = len(response.data[0].embedding)
        _print_result("Embeddings API", True, f"deployment={config.embedding_deployment}, dim={dim}")
        return True
    except Exception as exc:
        _print_result("Embeddings API", False, str(exc))
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Azure OpenAI API smoke tester")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--chat-only", action="store_true", help="Run only chat completion test")
    mode.add_argument("--embedding-only", action="store_true", help="Run only embedding test")

    parser.add_argument("--endpoint", help="Override AZURE_OPENAI_ENDPOINT")

    parser.add_argument("--chat-api-key", help="Override chat API key")
    parser.add_argument("--chat-deployment", help="Override chat deployment")
    parser.add_argument("--chat-api-version", help="Override chat API version")

    parser.add_argument("--embedding-api-key", help="Override embedding API key")
    parser.add_argument("--embedding-deployment", help="Override embedding deployment")
    parser.add_argument("--embedding-api-version", help="Override embedding API version")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        config = load_config(args)
    except Exception as exc:
        print(f"[FAIL] Config: {exc}")
        return 2

    run_chat = not args.embedding_only
    run_embeddings = not args.chat_only

    chat_ok = True
    emb_ok = True

    if run_chat:
        chat_ok = test_chat(config)
    if run_embeddings:
        emb_ok = test_embeddings(config)

    ok = chat_ok and emb_ok
    print("\nSummary:")
    print(f"  chat_test      : {'PASS' if chat_ok else 'FAIL'}")
    print(f"  embedding_test : {'PASS' if emb_ok else 'FAIL'}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
