import tempfile
from pathlib import Path

import streamlit as st

from rag.config import load_settings
from rag.indexer import build_index_from_dir, load_index
from rag.answerer import answer_question

APP_TITLE = "AI RAG Knowledge Assistant"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Upload documents, build a knowledge base, then ask questions grounded on retrieved context.")

settings = load_settings()

with st.sidebar:
    st.header("Knowledge Base")
    st.write("1) Upload docs  2) Build index  3) Ask questions")
    top_k = st.slider("Top-K retrieved chunks", 1, 10, 4, 1)
    use_llm = st.toggle("Use LLM answerer (needs API key)", value=bool(settings.openai_api_key))
    st.divider()
    st.subheader("Tips")
    st.write("- No API key? Works with local fallback based on retrieved context.")
    st.write("- Best results when documents are clear and specific.")

if "work_dir" not in st.session_state:
    st.session_state.work_dir = tempfile.mkdtemp(prefix="rag_kb_")
work_dir = Path(st.session_state.work_dir)

data_dir = work_dir / "data"
index_dir = work_dir / ".rag_index"
data_dir.mkdir(parents=True, exist_ok=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1) Upload Documents")
    uploaded = st.file_uploader("Upload .txt / .md files", type=["txt", "md"], accept_multiple_files=True)
    if uploaded:
        for uf in uploaded:
            (data_dir / uf.name).write_bytes(uf.getbuffer())
        st.success(f"Saved {len(uploaded)} file(s).")

    st.write("Current files:")
    files = sorted([p.name for p in data_dir.glob("*") if p.is_file()])
    if files:
        st.code("\n".join(files))
    else:
        st.info("No files yet. Upload to begin.")

    st.subheader("2) Build Index")
    if st.button("Build / Rebuild Index", type="primary"):
        if not files:
            st.error("Please upload at least one file first.")
        else:
            with st.spinner("Building FAISS index..."):
                build_index_from_dir(str(data_dir), str(index_dir), settings)
            st.success("Index built successfully.")

with col2:
    st.subheader("3) Ask a Question")
    question = st.text_input("Question", placeholder="e.g., What are the login requirements?")
    if st.button("Ask", disabled=not question):
        if not index_dir.exists():
            st.error("Index not found. Please build index first.")
        else:
            with st.spinner("Retrieving context and generating answer..."):
                idx = load_index(str(index_dir))
                result = answer_question(
                    question=question,
                    index=idx,
                    top_k=top_k,
                    settings=settings,
                    use_llm=use_llm,
                )

            st.markdown("### Answer")
            st.write(result["answer"])

            st.markdown("### Retrieved Context (Top-K)")
            for i, chunk in enumerate(result["contexts"], start=1):
                with st.expander(f"Chunk #{i} — {chunk['source']}"):
                    st.write(chunk["text"])

            st.markdown("### Debug Info")
            st.json({"used_llm": result["used_llm"], "top_k": top_k, "chunks": len(result["contexts"])})
