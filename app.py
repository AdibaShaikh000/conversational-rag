import os

import streamlit as st
from dotenv import load_dotenv

from src.chain import build_chain, make_history
from src.loader import load_pdf, load_pdf_bytes, split_text
from src.reranker import build_reranking_retriever
from src.vector_store import build_vector_store, get_base_retriever

load_dotenv()

DEFAULT_PDF = "resources/Approved IP Law.pdf"
MEMORY_WINDOW = 10  # number of messages (user + assistant) kept in context


# ── Cached pipeline ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading document and building index…")
def build_retriever(file_key: str, file_bytes: bytes | None, api_key: str):
    """
    Builds the FAISS vector store and wraps it with the cross-encoder reranker.
    Cached by (file_key, api_key) so the expensive embedding step runs once.
    """
    text = load_pdf_bytes(file_bytes) if file_bytes else load_pdf(DEFAULT_PDF)
    chunks = split_text(text)
    vs = build_vector_store(chunks, api_key)
    base_ret = get_base_retriever(vs, k=20)
    return build_reranking_retriever(base_ret, top_n=5)


# ── Session helpers ───────────────────────────────────────────────────────────

def reset_session():
    for key in ("rag_chain", "messages"):
        st.session_state.pop(key, None)


def get_api_key() -> str:
    return st.session_state.get("api_key") or os.getenv("OPENAI_API_KEY", "")


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="RAG Q&A Assistant",
        page_icon="⚖️",
        layout="wide",
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("RAG Q&A")
        st.caption("Cross-Encoder Reranking · Conversational Memory")
        st.divider()

        st.subheader("OpenAI API Key")
        key_input = st.text_input(
            "API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            label_visibility="collapsed",
            placeholder="sk-…",
        )
        if key_input:
            st.session_state["api_key"] = key_input

        st.divider()
        st.subheader("Document")
        st.caption(f"Default: `{DEFAULT_PDF}`")
        uploaded = st.file_uploader("Upload a PDF", type="pdf", label_visibility="collapsed")

        file_id = getattr(uploaded, "file_id", "default")
        if file_id != st.session_state.get("_file_id"):
            reset_session()
            st.session_state["_file_id"] = file_id

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.pop("messages", None)
            st.session_state.pop("rag_chain", None)
            st.rerun()

    # ── Guard: require API key ─────────────────────────────────────────────────
    api_key = get_api_key()
    if not api_key:
        st.info("Enter your OpenAI API key in the sidebar to get started.")
        st.stop()

    # ── Build retriever & chain (cached / session) ────────────────────────────
    file_bytes = uploaded.getvalue() if uploaded else None
    retriever = build_retriever(file_id, file_bytes, api_key)

    if "rag_chain" not in st.session_state:
        st.session_state["rag_chain"] = build_chain(retriever, api_key)
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # ── Header ────────────────────────────────────────────────────────────────
    doc_name = uploaded.name if uploaded else "Approved IP Law"
    st.title(f"⚖️ {doc_name}")
    st.caption(
        "Retrieval pipeline: **FAISS** (k=20) → **Cross-Encoder Reranker** (top 5) → **GPT-4o-mini**"
    )

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Retrieved sources"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**Chunk {i}**")
                        st.markdown(f"> {src[:400]}…")

    # ── Chat input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a question about your document…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build sliding-window history for the chain (exclude current message)
        history_window = st.session_state.messages[-(MEMORY_WINDOW + 1):-1]
        chat_history = make_history(history_window)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = st.session_state.rag_chain.invoke(
                    {"question": prompt, "chat_history": chat_history}
                )

            answer = result["answer"]
            sources = [d.page_content for d in result.get("source_documents", [])]

            st.markdown(answer)
            if sources:
                with st.expander("Retrieved sources"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**Chunk {i}**")
                        st.markdown(f"> {src[:400]}…")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    main()
