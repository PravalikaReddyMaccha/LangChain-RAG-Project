"""Quick end-to-end demo: ingest the sample handbook and ask two questions.

Run from the project root:  python main.py
For everyday use, see cli.py (terminal) and app.py (Streamlit).
"""

import logging

from src.library import build_library

logging.getLogger("pypdf").setLevel(logging.ERROR)

SAMPLE_FILE = "data/sample.md"
QUESTIONS = [
    "How many days of paid leave do employees accrue per month?",
    "What is the best pancake recipe?",
]


def main() -> None:
    library = build_library()
    result = library.ingest_file(SAMPLE_FILE)
    print(f"{result.status} {result.source}")

    for question in QUESTIONS:
        answer = library.ask(question)
        print(f"\nQ: {question}\nA: {answer.text}")
        for hit in answer.sources:
            print(f"   - {hit.chunk.source} (score {hit.score:.2f})")


if __name__ == "__main__":
    main()
