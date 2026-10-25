---
title: Chat With Your Documents
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Chat with your documents (LangChain RAG)

A retrieval-augmented generation (RAG) app: add PDF, Markdown or text files, then ask questions and get answers grounded in those files, with citations.

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, run locally (no API key needed)
- **Vector store:** FAISS with exact cosine search, saved to disk
- **LLM:** Groq (`openai/gpt-oss-20b`), instructed to answer only from the retrieved context
- **Interfaces:** a command-line tool and a Streamlit web app

## How it works

```text
INGEST   file → load (PDF keeps page numbers) → split (800 chars, 120 overlap)
              → embed → FAISS index on disk

ASK      question → embed → top 4 chunks by cosine similarity → drop weak matches
              → prompt with labelled context → Groq → answer with [p. N] / [file] citations
```

## Project structure

```text
src/
  models.py        DocumentChunk, RetrievedChunk, Answer, IngestResult (Pydantic)
  ingestion.py     load files and split them into chunks
  embeddings.py    embedding model, loaded once
  vector_store.py  FAISS store: add, search, list, delete by file
  llm.py           Groq chat model
  rag.py           retrieve + prompt + generate
  library.py       DocumentLibrary: the single entry point used by both interfaces
cli.py             command-line interface
app.py             Streamlit app
main.py            small end-to-end demo
data/sample.md     sample document
```

## Run locally

Requires Python 3.12 and a free [Groq API key](https://console.groq.com/keys).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root (it is git-ignored):

```text
GROQ_API_KEY=your_key_here
```

The first run downloads the embedding model (about 90 MB).

### Command line

```bash
python cli.py ingest data/sample.md            # add files
python cli.py ingest --folder data --replace   # add a folder, re-ingesting changed files
python cli.py sources                          # list stored files
python cli.py ask                              # interactive questions (Ctrl+C to quit)
python cli.py delete sample.md                 # remove a file
```

### Web app

```bash
python -m streamlit run app.py
```

Upload files in the sidebar, click **Ingest**, then ask questions in the chat.

## Deploying to Hugging Face Spaces

1. Create a Space with the **Docker** SDK (blank template).
2. In the Space's **Settings → Variables and secrets**, add a secret named `GROQ_API_KEY`.
3. Push this repository to the Space; the `Dockerfile` builds and starts the app on port 7860.

## Limitations

- **Shared document store.** Every visitor of a deployed app sees, uses and can delete the same documents, and all questions use your Groq quota. Keep a public deployment private, or add per-user stores.
- **No persistence on Spaces.** Uploaded documents are lost when the Space restarts.
- **Files are identified by name.** Re-ingest an edited file with `--replace` (or the *Replace* checkbox).
- Chroma was replaced by FAISS because Chroma 1.5.9's approximate index missed newly added chunks ([chroma-core/chroma#7758](https://github.com/chroma-core/chroma/issues/7758)).
