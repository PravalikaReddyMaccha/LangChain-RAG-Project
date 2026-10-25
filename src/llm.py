from functools import lru_cache

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

LLM_MODEL = "openai/gpt-oss-20b"


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """Create the Groq chat model once; later calls return the same instance."""
    return ChatGroq(
        model=LLM_MODEL,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )