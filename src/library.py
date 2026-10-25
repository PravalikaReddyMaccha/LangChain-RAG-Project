from pathlib import Path

from src.ingestion import SUPPORTED_EXTENSIONS, DocumentIngestionService
from src.models import Answer, IngestResult
from src.rag import RAGService
from src.vector_store import VectorStore


class DocumentLibrary:
    """One entry point for the CLI and the app: ingest, list, delete and ask."""

    def __init__(self, store: VectorStore, ingestion: DocumentIngestionService, rag: RAGService) -> None:
        self._store = store
        self._ingestion = ingestion
        self._rag = rag

    def ingest_file(self, file_path: str | Path, replace: bool = False) -> IngestResult:
        """Add one file. Skips it if already stored, unless replace=True."""
        name = Path(file_path).name

        if not Path(file_path).is_file():
            return IngestResult(source=name, status="failed", message=f"No such file: {file_path}")

        already = self._store.has_source(name)
        if already and not replace:
            return IngestResult(source=name, status="skipped")

        try:
            chunks = self._ingestion.parse_and_chunk(file_path)
        except (FileNotFoundError, ValueError) as error:
            return IngestResult(source=name, status="failed", message=str(error))

        if not chunks:
            return IngestResult(source=name, status="empty")

        # Delete the old version only now that the new chunks are ready.
        if already:
            self._store.delete_source(name)
        self._store.add_chunks(chunks)

        return IngestResult(source=name, status="replaced" if already else "added", chunks=len(chunks))

    def ingest_folder(self, folder: str | Path, replace: bool = False) -> list[IngestResult]:
        """Ingest every supported file directly inside a folder."""
        folder = Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a folder: {folder}")

        results: list[IngestResult] = []
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                results.append(self.ingest_file(path, replace=replace))
        return results

    def list_sources(self) -> list[str]:
        """Names of all stored files."""
        return self._store.list_sources()

    def delete(self, source: str) -> None:
        """Remove one file's chunks from the store."""
        self._store.delete_source(source)

    def ask(self, question: str) -> Answer:
        """Answer a question from the stored documents."""
        return self._rag.answer(question)


def build_library(chunk_size: int = 800, chunk_overlap: int = 120) -> DocumentLibrary:
    """Create a DocumentLibrary with the default store, ingestion settings and RAG service."""
    store = VectorStore()
    return DocumentLibrary(
        store,
        DocumentIngestionService(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        RAGService(store),
    )

