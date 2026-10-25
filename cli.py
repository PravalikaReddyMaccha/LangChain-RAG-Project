"""Command-line interface for the document Q&A app. Run from the project root."""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import logging

from src.library import DocumentLibrary, build_library
from src.models import IngestResult
from src.rag import NOT_FOUND

logging.getLogger("pypdf").setLevel(logging.ERROR)


def cmd_ask(library: DocumentLibrary) -> None:
    """Interactive question loop. Ends on 'exit', an empty line, or Ctrl+C."""
    print("Type a question, or 'exit' to quit.")
    try:
        while True:
            question = input("\nYou: ").strip()
            if question in ("", "exit", "quit"):
                break
            try:
                answer = library.ask(question)
            except Exception as error:
                print(f"Error: {error}")
                continue
            print("AI:", answer.text)

            if answer.text == NOT_FOUND:
                continue
            for hit in answer.sources:
                page = hit.chunk.metadata.get("page_number")
                if page is None:
                    print(f"   source: {hit.chunk.source}")
                else:
                    print(f"   source: {hit.chunk.source} p. {page}")
    except KeyboardInterrupt:
        pass
    print("\nBye")


def cmd_sources(library: DocumentLibrary) -> None:
    """Print every stored file name."""
    sources = library.list_sources()
    if not sources:
        print("No documents stored yet.")
        return
    for name in sources:
        print(f"  {name}")


def print_result(result: IngestResult) -> None:
    """Print one line describing what happened to one file."""
    line = f"{result.status:<9} {result.source}"
    if result.chunks:
        line += f" ({result.chunks} chunks)"
    if result.message:
        line += f": {result.message}"
    print(line)


def cmd_ingest(library: DocumentLibrary, files: list[str], folder: str | None, replace: bool) -> None:
    """Ingest the given files and/or every supported file in a folder."""
    if not files and folder is None:
        print("Nothing to ingest. Give file names or --folder.")
        return

    if folder is not None:
        try:
            for result in library.ingest_folder(folder, replace=replace):
                print_result(result)
        except NotADirectoryError as error:
            print(error)

    for path in files:
        print_result(library.ingest_file(path, replace=replace))


def cmd_delete(library: DocumentLibrary, source: str) -> None:
    """Remove one stored file and all its chunks."""
    if source not in library.list_sources():
        print(f"Not found: {source}")
        return
    library.delete(source)
    print(f"Deleted {source}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask questions about your documents.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ask", help="start an interactive question loop")
    commands.add_parser("sources", help="list stored files")

    ingest = commands.add_parser("ingest", help="add files or a folder to the store")
    ingest.add_argument("files", nargs="*", help="files to ingest")
    ingest.add_argument("--folder", help="ingest every supported file in this folder")
    ingest.add_argument("--replace", action="store_true", help="re-ingest files that are already stored")

    delete = commands.add_parser("delete", help="remove a stored file")
    delete.add_argument("source", help="file name as shown by 'sources'")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    library = build_library()

    if args.command == "ask":
        cmd_ask(library)
    elif args.command == "sources":
        cmd_sources(library)
    elif args.command == "ingest":
        cmd_ingest(library, args.files, args.folder, args.replace)
    elif args.command == "delete":
        cmd_delete(library, args.source)


if __name__ == "__main__":
    main()