from __future__ import annotations

from collections.abc import Iterator

import ollama
from openai import OpenAI

from config import LLM_PROVIDER, OLLAMA_MODEL, OPENAI_API_KEY, OPENAI_MODEL


def _stream_with_ollama(prompt: str, system_prompt: str) -> Iterator[str]:
    stream = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield content


def _stream_with_openai(prompt: str, system_prompt: str) -> Iterator[str]:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    stream = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        if content:
            yield content


def call_llm(prompt: str, system_prompt: str, stream: bool = True):
    if LLM_PROVIDER == "ollama":
        if stream:
            return _stream_with_ollama(prompt, system_prompt)
        return "".join(_stream_with_ollama(prompt, system_prompt))

    if LLM_PROVIDER == "openai":
        if stream:
            return _stream_with_openai(prompt, system_prompt)
        return "".join(_stream_with_openai(prompt, system_prompt))

    raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")
