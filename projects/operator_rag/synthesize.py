"""Answer synthesis over retrieved chunks, through an OpenAI-compatible endpoint.

The endpoint URL comes from OPERATOR_RAG_LLM_URL and nothing else is read from the
environment, so no credential ever has to live in the repo. The call is a plain
HTTP POST; the tests replace `post_chat` and never touch the network.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable

DEFAULT_ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"

SYSTEM_PROMPT = (
    "You answer only from the numbered study chunks you are given. "
    "Reply with one JSON object and nothing else, with exactly these keys: "
    '"answer" (a short string), "cited_chunks" (a list of the chunk numbers you '
    'actually used), and "refused" (a boolean). '
    "Set refused to true and cited_chunks to [] when the question asks for live "
    "operating advice, emergency action, bypass or override steps, or anything the "
    "chunks do not contain. Never invent a source and never add a chunk you did not use."
)


def post_chat(endpoint: str, payload: dict) -> dict:
    """POST a chat completion and return the parsed JSON body. No auth header is sent."""
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def synthesize(
    question: str,
    retrieved: list[dict],
    *,
    endpoint: str | None = None,
    poster: Callable[[str, dict], dict] = post_chat,
) -> dict:
    """Return the grounded answer plus the raw model output it was grounded from.

    `cited_chunks` holds the text of each chunk the model cited by a real number.
    `raw_cited` holds the citation list exactly as the model returned it, including
    numbers that name no retrieved chunk, so a later check can see what grounding
    dropped. `raw_response` is the model message unchanged.
    """
    numbered = [
        {"n": index, "source": chunk.get("source", ""), "title": chunk.get("title", ""),
         "text": chunk.get("text", "")}
        for index, chunk in enumerate(retrieved, start=1)
    ]
    payload = {
        "model": "local",
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(
                {"question": question, "chunks": numbered}, ensure_ascii=False)},
        ],
    }
    url = endpoint or os.environ.get("OPERATOR_RAG_LLM_URL", DEFAULT_ENDPOINT)
    reply = poster(url, payload)
    return _ground(_parse(reply), numbered, reply)


def _parse(reply: dict) -> dict:
    content = reply["choices"][0]["message"]["content"]
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model reply contained no JSON object")
    parsed = json.loads(content[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("model reply was not a JSON object")
    return parsed


def _ground(parsed: dict, numbered: list[dict], reply: dict) -> dict:
    by_number = {chunk["n"]: chunk for chunk in numbered}
    raw = parsed.get("cited_chunks") or []
    cited = [by_number[n]["text"] for n in raw if isinstance(n, int) and n in by_number]
    kept = {"answer": str(parsed.get("answer", "")).strip(), "cited_chunks": cited, "refused": False,
            "raw_cited": list(raw), "raw_response": reply["choices"][0]["message"]["content"]}
    if bool(parsed.get("refused")) or not cited:
        return {**kept, "answer": "The available study material does not support an answer to that question.",
                "cited_chunks": [], "refused": True}
    return kept
