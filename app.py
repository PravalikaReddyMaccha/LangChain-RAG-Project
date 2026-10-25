"""Streamlit chat app. Run from the project root: python -m streamlit run app.py"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import logging
from pathlib import Path

import streamlit as st

from src.library import DocumentLibrary, build_library
from src.models import Answer, IngestResult
from src.rag import NOT_FOUND

logging.getLogger("pypdf").setLevel(logging.ERROR)
st.set_page_config(page_title="Chat with your documents", page_icon="📚")

UPLOAD_DIR = Path("uploads")


@st.cache_resource(show_spinner="Loading your documents...")
def get_library() -> DocumentLibrary:
    """Build the library once; every rerun and every tab reuses it."""
    return build_library()


def upload_files(library: DocumentLibrary) -> None:
    """Let the user upload files and ingest them into the store."""
    st.header("⬆️ Add documents")
    files = st.file_uploader(
        "PDF, Markdown or text files",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )
    replace = st.checkbox("Replace if already stored")

    if files and st.button("Ingest"):
        UPLOAD_DIR.mkdir(exist_ok=True)
        for uploaded in files:
            path = UPLOAD_DIR / uploaded.name
            path.write_bytes(uploaded.getvalue())
            with st.spinner(f"Ingesting {uploaded.name}... (don't click anything until it finishes)"):
                result = library.ingest_file(path, replace=replace)
            show_result(result)


def show_result(result: IngestResult) -> None:
    """Show one coloured message describing what happened to one file."""
    if result.status in ("added", "replaced"):
        st.success(f"{result.status} {result.source} ({result.chunks} chunks)")
    elif result.status == "skipped":
        st.info(f"skipped {result.source} (already stored)")
    elif result.status == "empty":
        st.warning(f"empty {result.source} (no text found)")
    else:
        st.error(f"failed {result.source}: {result.message}")


def source_labels(answer: Answer) -> list[str]:
    """One short label per retrieved chunk, e.g. 'book.pdf p. 115' or 'sample.md'."""
    if answer.text == NOT_FOUND:
        return []
    labels = []
    for hit in answer.sources:
        page = hit.chunk.metadata.get("page_number")
        if page is not None:
            labels.append(f"{hit.chunk.source} p. {page}")
        else:
            labels.append(hit.chunk.source)
    return labels


def show_sources(labels: list[str]) -> None:
    """Show source labels in a collapsed section, so long file names don't fill the chat."""
    if labels:
        with st.expander(f"Sources ({len(labels)})"):
            for label in labels:
                st.caption(label)


def show_sidebar(library: DocumentLibrary) -> None:
    """Upload box, then the list of stored documents with a delete button each."""
    with st.sidebar:
        upload_files(library)
        st.divider()
        sources = library.list_sources()
        st.header(f"📄 Documents ({len(sources)})")

        if not sources:
            st.info("No documents yet.")
            return

        for name in sources:
            name_col, button_col = st.columns([5, 1])
            name_col.write(name)
            if button_col.button("🗑", key=f"delete-{name}", help=f"Delete {name}"):
                library.delete(name)
                st.rerun()


def main() -> None:
    st.title("📚 Chat with your documents")
    library = get_library()
    show_sidebar(library)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            show_sources(message.get("sources", []))

    question = st.chat_input("Ask a question about your documents")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = library.ask(question)
                except Exception as error:
                    st.error(f"Couldn't get an answer: {error}")
                    return
            st.markdown(answer.text)
            labels = source_labels(answer)
            show_sources(labels)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer.text, "sources": labels}
        )


if __name__ == "__main__":
    main()