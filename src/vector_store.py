from pathlib import Path

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from src.embeddings import get_embeddings
from src.models import DocumentChunk, RetrievedChunk

DEFAULT_INDEX_DIR = Path("faiss_index")


class VectorStore:
    """Stores chunk embeddings on disk and finds the most similar ones.

    Backed by FAISS with an exact inner-product index. Embeddings are normalized,
    so the inner product equals cosine similarity and search never misses a chunk.
    (Chroma 1.5.9 was dropped: its approximate index missed newly added chunks,
    see https://github.com/chroma-core/chroma/issues/7758.)
    """

    def __init__(self, index_dir: str | Path = DEFAULT_INDEX_DIR) -> None:
        self._index_dir = Path(index_dir)
        embeddings = get_embeddings()

        if (self._index_dir / "index.faiss").exists():
            self._store = FAISS.load_local(
                str(self._index_dir),
                embeddings,
                # Loading uses pickle. Safe here because only this app writes the folder.
                allow_dangerous_deserialization=True,
                # Not saved with the index, so it must be passed again on every load.
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )
        else:
            dimension = len(embeddings.embed_query("dimension check"))
            self._store = FAISS(
                embedding_function=embeddings,
                index=faiss.IndexFlatIP(dimension),
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )

    def add_chunks(self, chunks: list[DocumentChunk], batch_size: int = 256) -> int:
        """Embed and store chunks. Adding a chunk whose id already exists overwrites it."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if not chunks:
            return 0

        # FAISS has no upsert: adding an existing id raises, so remove old copies first.
        stored_ids = set(self._store.index_to_docstore_id.values())
        existing = [chunk.chunk_id for chunk in chunks if chunk.chunk_id in stored_ids]
        if existing:
            self._store.delete(ids=existing)

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self._store.add_texts(
                texts=[chunk.text for chunk in batch],
                metadatas=[self._to_metadata(chunk) for chunk in batch],
                ids=[chunk.chunk_id for chunk in batch],
            )

        self._save()
        return len(chunks)

    def search(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        """Return the k stored chunks most similar to the query, best first."""
        if not query.strip() or self.count() == 0:
            return []

        # Raw inner-product score = cosine similarity (higher is better). Avoid
        # similarity_search_with_relevance_scores: for inner product it returns
        # 1 - similarity, which would invert the scores.
        results = self._store.similarity_search_with_score(query, k=k)
        return [
            RetrievedChunk(chunk=self._to_chunk(doc), score=float(score))
            for doc, score in results
        ]

    def count(self) -> int:
        """Number of chunks currently stored."""
        return self._store.index.ntotal

    def has_source(self, source: str) -> bool:
        """True if at least one chunk from this file is stored."""
        return any(doc.metadata.get("source") == source for doc in self._all_documents())

    def delete_source(self, source: str) -> None:
        """Remove every stored chunk that came from this file."""
        ids = [doc.id for doc in self._all_documents() if doc.metadata.get("source") == source]
        if ids:
            self._store.delete(ids=ids)
            self._save()

    def list_sources(self) -> list[str]:
        """Names of all stored files.

        Scans every chunk, which is fine for thousands of chunks. A larger system
        would keep a separate source -> ids index instead.
        """
        sources = set()
        for doc in self._all_documents():
            source = doc.metadata.get("source")
            if source is not None:
                sources.add(source)
        return sorted(sources)

    def _all_documents(self) -> list[Document]:
        """Every stored chunk as a LangChain Document (text, metadata, id)."""
        return self._store.get_by_ids(list(self._store.index_to_docstore_id.values()))

    def _save(self) -> None:
        """Write the index to disk. FAISS keeps everything in memory otherwise."""
        self._store.save_local(str(self._index_dir))

    @staticmethod
    def _to_metadata(chunk: DocumentChunk) -> dict:
        """Convert a chunk's metadata into the form stored alongside its vector."""
        metadata = {key: value for key, value in chunk.metadata.items() if value is not None}
        metadata["source"] = chunk.source
        return metadata

    @staticmethod
    def _to_chunk(doc: Document) -> DocumentChunk:
        """Rebuild a DocumentChunk from a stored Document. Reverses _to_metadata."""
        metadata = dict(doc.metadata)
        source = metadata.pop("source", "unknown")
        return DocumentChunk(
            chunk_id=doc.id or "",
            text=doc.page_content,
            source=source,
            metadata=metadata,
        )
