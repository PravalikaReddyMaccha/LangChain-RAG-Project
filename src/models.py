from pydantic import BaseModel,Field, ConfigDict
from typing import Literal


class DocumentChunk(BaseModel):
    """A single contiguous slice of a source document, ready for embedding."""
    model_config = ConfigDict(frozen = True)
    chunk_id: str
    text: str 
    source: str
    metadata: dict = Field(default_factory = dict, description = "Metadata of the documents")

class RetrievedChunk(BaseModel):
    """A chunk returned by a search, with its relevance score."""

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk
    score: float

class Answer(BaseModel):
    """An LLM-generated answer together with the chunks it was based on."""

    model_config = ConfigDict(frozen=True)

    question: str
    text: str
    sources: list[RetrievedChunk]

class IngestResult(BaseModel):
    """What happened when one file was ingested."""

    model_config = ConfigDict(frozen=True)

    source: str
    status: Literal["added","replaced","skipped","empty","failed"]
    chunks: int = 0
    message: str = ""


