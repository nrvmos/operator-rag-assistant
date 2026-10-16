from __future__ import annotations

from pathlib import Path

import streamlit as st

from industrial_ai_portfolio.rag import (
    build_study_answer,
    chunk_documents,
    is_unsafe_live_operations_request,
    load_markdown_documents,
    retrieve_chunks,
)
from projects.operator_rag.synthesize import synthesize


st.set_page_config(page_title="Operator Knowledge RAG", layout="wide")

st.title("Operator Knowledge RAG Assistant")
st.caption("Citation-first study assistant. Not for live operations or site-specific procedures.")

doc_dir = Path(__file__).parent / "documents"
documents = load_markdown_documents(doc_dir)
chunks = chunk_documents(documents)

with st.sidebar:
    st.header("Corpus")
    st.metric("Documents", len(documents))
    st.metric("Chunks", len(chunks))
    st.write("Sources")
    for document in documents:
        st.write(f"- {document.title}")

question = st.text_input(
    "Ask a study question",
    value="What makes a good shift handover for abnormal equipment conditions?",
)

top_k = st.slider("Sources to retrieve", min_value=1, max_value=5, value=3)

if question:
    retrieved = retrieve_chunks(question, chunks, top_k=top_k)

    if is_unsafe_live_operations_request(question):
        st.error(build_study_answer(question, retrieved))
    else:
        answer = synthesize(question, retrieved)
        st.subheader("Study Answer")
        st.write(answer["answer"])
        if answer["refused"]:
            st.warning("No supporting source in the study documents, so nothing is cited.")

    st.subheader("Retrieved Sources")
    for index, result in enumerate(retrieved, start=1):
        with st.expander(f"{index}. {result['title']} - {result['source']} (score {result['score']:.2f})"):
            st.write(result["text"])

st.divider()
st.write(
    "Safety boundary: this demo refuses live operating advice. Use approved site procedures, "
    "qualified supervision, and emergency protocols for real conditions."
)
