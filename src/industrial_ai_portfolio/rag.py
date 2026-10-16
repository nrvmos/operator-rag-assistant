from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REFUSAL = (
    "I cannot advise on live operations, emergency response, bypassing safeguards, "
    "or site-specific procedures. Use approved procedures, qualified supervision, "
    "and emergency protocols. I can help summarize public or simulated study material."
)

NO_MATCH = "I could not find a matching excerpt in the available study documents."

# These rules cover demo examples; they are not a validated safety classifier.
REQUEST = (
    r"\b(?:how (?:do|can|should|would) (?:i|we)|how to|what should (?:i|we) do|"
    r"(?:can|could|should|may) (?:i|we)|which valve|give me|tell me|walk me|"
    r"steps|instructions|write|create|generate|recommend)\b"
)
ACTION = (
    r"\b(?:start|startup|restart|stop|shutdown|shut down|isolate|open|close|"
    r"operate|reset|adjust|setpoint|valve lineup|procedure|sop)\b"
)
SITE_CONTEXT = (
    r"\b(?:(?:my|our|this) (?:plant|site|unit|pump|compressor|boiler|reactor|"
    r"valve|equipment)|site specific|plant specific)\b|\b[a-z]{1,4}-\d+\b"
)
LIVE_CONTEXT = (
    r"\b(?:right now|live (?:plant|operations?|equipment)|active upset|"
    r"on shift|currently (?:running|operating)|happening now|emergency)\b"
)
SAFEGUARD_ACTION = r"\b(?:bypass|override|defeat|disable|circumvent)\b"
PROCEDURE_REPLACEMENT = (
    r"\b(?:replace|supersede|ignore|skip|instead of|substitute for|without following)\b"
    r".{0,100}\b(?:procedure|sop|manual|supervisor|supervision)\b"
)


@dataclass(frozen=True)
class Document:
    source: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    source: str
    title: str
    text: str


def is_unsafe_live_operations_request(question: str) -> bool:
    """Recognize selected operating requests without banning study vocabulary."""
    lowered = " ".join(question.lower().replace("site-specific", "site specific").split())
    asks_for_action = bool(re.search(REQUEST, lowered))
    return bool(
        re.search(PROCEDURE_REPLACEMENT, lowered)
        or (re.search(SAFEGUARD_ACTION, lowered) and (
            asks_for_action or re.match(r"(?:please )?" + SAFEGUARD_ACTION, lowered)
        ))
        or (asks_for_action and re.search(SITE_CONTEXT, lowered) and re.search(ACTION, lowered))
        or (asks_for_action and re.search(LIVE_CONTEXT, lowered))
        or re.search(r"\bwhat should (?:i|we) do\b", lowered)
        or re.search(r"\bwhich valve\b|\bvalve lineup\b", lowered)
    )


def load_markdown_documents(directory: Path) -> list[Document]:
    documents = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = _extract_title(text) or path.stem.replace("_", " ").title()
        documents.append(Document(source=path.name, title=title, text=text))
    return documents


def chunk_documents(documents: list[Document], chunk_size: int = 140, overlap: int = 30) -> list[Chunk]:
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size.")
    chunks: list[Chunk] = []
    step = chunk_size - overlap

    for document in documents:
        body = " ".join(line for line in document.text.splitlines() if not re.match(r"^\s*#+\s", line))
        words = body.split()
        for start in range(0, len(words), step):
            window = words[start : start + chunk_size]
            chunks.append(Chunk(source=document.source, title=document.title, text=" ".join(window)))
            if start + chunk_size >= len(words):
                break

    return chunks


def retrieve_chunks(question: str, chunks: list[Chunk], top_k: int = 3) -> list[dict[str, object]]:
    if not question.strip() or not chunks or top_k <= 0:
        return []

    corpus = [chunk.text for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    analyze = vectorizer.build_analyzer()
    if not any(analyze(text) for text in corpus) or not analyze(question):
        return []
    matrix = vectorizer.fit_transform(corpus)
    scores = cosine_similarity(vectorizer.transform([question]), matrix).flatten()
    ranked_indexes = sorted(range(len(chunks)), key=lambda index: (-scores[index], index))[:top_k]

    results = []
    for index in ranked_indexes:
        if scores[index] <= 0:
            continue
        chunk = chunks[index]
        results.append(
            {
                "score": float(scores[index]),
                "source": chunk.source,
                "title": chunk.title,
                "text": chunk.text,
            }
        )

    return results


def build_study_answer(question: str, retrieved: list[dict[str, object]]) -> str:
    if is_unsafe_live_operations_request(question):
        return REFUSAL

    analyze = TfidfVectorizer(stop_words="english").build_analyzer()
    question_terms = set(analyze(question))
    excerpts = []
    seen = set()
    for result in retrieved:
        sentences = re.split(r"(?<=[.!?])\s+", str(result["text"]).strip())
        ranked = sorted(
            sentences,
            key=lambda sentence: len(question_terms.intersection(analyze(sentence))),
            reverse=True,
        )
        for sentence in ranked:
            if sentence in seen or not question_terms.intersection(analyze(sentence)):
                continue
            number = len(excerpts) + 1
            excerpts.append(f"> {sentence}\n\n[{number}] {result['title']} ({result['source']})")
            seen.add(sentence)
            break
        if len(excerpts) == 3:
            break

    if not excerpts:
        return NO_MATCH
    return (
        "Study answer — excerpts from the retrieved material:\n\n"
        + "\n\n".join(excerpts)
        + "\n\nThese excerpts may only partially address your question. Check the source context below."
    )


def _extract_title(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None
