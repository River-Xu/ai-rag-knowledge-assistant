from __future__ import annotations

import argparse

from .config import load_settings
from .indexer import build_index_from_dir, load_index
from .answerer import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag-cli", description="RAG Knowledge Assistant CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Build index from documents")
    p_index.add_argument("--data_dir", required=True)
    p_index.add_argument("--index_dir", required=True)

    p_ask = sub.add_parser("ask", help="Ask question from an index")
    p_ask.add_argument("--index_dir", required=True)
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--top_k", type=int, default=4)
    p_ask.add_argument("--use_llm", action="store_true")

    args = parser.parse_args()
    settings = load_settings()

    if args.cmd == "index":
        build_index_from_dir(args.data_dir, args.index_dir, settings=settings)
        print(f"Index built at: {args.index_dir}")
        return

    if args.cmd == "ask":
        idx = load_index(args.index_dir)
        res = answer_question(
            question=args.question,
            index=idx,
            top_k=args.top_k,
            settings=settings,
            use_llm=args.use_llm,
        )
        print(res["answer"])
        return


if __name__ == "__main__":
    main()
