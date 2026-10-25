from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models import DocumentChunk

_LOADERS = {
    ".pdf": lambda path: PyPDFLoader(str(path)),
    ".md": lambda path: TextLoader(str(path), encoding="utf-8"),
    ".txt": lambda path: TextLoader(str(path), encoding="utf-8"),
}
SUPPORTED_EXTENSIONS = set(_LOADERS)


class DocumentIngestionService:
    """Turns a file on disk into a list of overlapping, metadata-rich chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than "
                f"chunk_size ({chunk_size}); otherwise the window never advances."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def parse_and_chunk(self, file_path: str | Path) -> list[DocumentChunk]:
        """Load a file, split it into chunks and wrap each one in a DocumentChunk."""
        path = Path(file_path)
        pieces = self._make_splitter().split_documents(self.load_file(path))
        total = len(pieces)

        chunks: list[DocumentChunk] = []
        for index, piece in enumerate(pieces):
            metadata = {
                "file_type": path.suffix.lower().lstrip("."),
                "chunk_index": index,
                "total_chunks": total,
                "start_index": piece.metadata.get("start_index"),
            }
            if "page" in piece.metadata:
                metadata["page_number"] = piece.metadata["page"] + 1

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{path.name}_chunk_{index}",
                    text=piece.page_content,
                    source=path.name,
                    metadata=metadata,
                )
            )
        return chunks

    def load_file(self, file_path: str | Path) -> list[Document]:
        """Read a file and return one Document per page (PDF) or one for the whole file."""
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(f"No such file: {path}")

        suffix = path.suffix.lower()
        make_loader = _LOADERS.get(suffix)
        if make_loader is None:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported file type {suffix!r} for {path.name}. Supported: {supported}"
            )

        return make_loader(path).load()

    def _make_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
