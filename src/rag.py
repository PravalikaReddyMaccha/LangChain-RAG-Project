from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.llm import get_llm
from src.models import Answer, RetrievedChunk
from src.vector_store import VectorStore

NOT_FOUND = "I couldn't find that in the documents."

SYSTEM_PROMPT = f"""You answer questions using ONLY the context provided from the user's documents.

Rules:
- If the context does not contain the answer, reply exactly: "{NOT_FOUND}"
- Do not use outside knowledge, even if you know the answer.
- After each fact, cite where it came from: the page like [p. 115], or the file name like [sample.md] if the source has no page.
- Be concise and clear."""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


class RAGService:
    """Answers questions using retrieved chunks as the only source of truth."""

    def __init__(self, store: VectorStore, k: int = 4, min_score: float = 0.3) -> None:
        self._store = store
        self._k = k
        self._min_score = min_score
        self._chain = PROMPT | get_llm() | StrOutputParser()

    def answer(self, question: str) -> Answer:
        """Retrieve relevant chunks, then ask the LLM to answer from them only."""
        results = self._retrieve(question)
        if not results:
            return Answer(question=question, text=NOT_FOUND, sources=[])

        text = self._chain.invoke(
            {"context": self._format_context(results), "question": question}
        )
        return Answer(question=question, text=text, sources=results)

    def _retrieve(self, question: str) -> list[RetrievedChunk]:
        """Search the store and drop results below the relevance threshold."""
        return [
            result
            for result in self._store.search(question, k=self._k)
            if result.score >= self._min_score
        ]

    @staticmethod
    def _format_context(results: list[RetrievedChunk]) -> str:
        """Label each chunk with its source (and page, if any) so the LLM can cite it."""
        blocks = []
        for result in results:
            page = result.chunk.metadata.get("page_number")
            if page is None:
                label = f"[{result.chunk.source}]"
            else:
                label = f"[{result.chunk.source}, p. {page}]"
            blocks.append(f"{label}\n{result.chunk.text}")
        return "\n\n---\n\n".join(blocks)