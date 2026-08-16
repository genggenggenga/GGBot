"""Unified entry point for versioned evaluation suites."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
from dataclasses import asdict
from datetime import datetime

from evaluation.local_eval_runner import run_local_eval, write_json_report, write_markdown_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("smoke", "golden", "bad_cases"),
        default="smoke",
    )
    parser.add_argument(
        "--mode",
        choices=("deterministic", "realistic"),
        default="deterministic",
    )
    parser.add_argument(
        "--rag-mode",
        choices=("dense", "hybrid", "rerank"),
        default="hybrid",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="报告输出目录（默认 data/eval/reports/<日期>）",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the report without writing JSON or Markdown artifacts.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Enable the optional LLM-as-Judge quality signal.",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    if args.mode == "realistic":
        if os.getenv("GGBOT_EVAL_REALISTIC_ENABLED", "").lower() not in {
            "1", "true", "yes", "on",
        }:
            raise RuntimeError(
                "realistic evaluation requires GGBOT_EVAL_REALISTIC_ENABLED=true "
                "and an environment-specific harness with real runtime components"
            )
        raise RuntimeError(
            "this repository does not bundle credentials or services for realistic "
            "evaluation; inject the deployed runtime harness instead"
        )

    judge = None
    if args.judge:
        from evaluation.judge import LLMJudge

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("--judge requires ANTHROPIC_API_KEY")
        judge = LLMJudge.from_api_key(
            api_key,
            os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            os.getenv("ANTHROPIC_BASE_URL") or None,
        )

    report = await run_local_eval(
        suite=args.suite,
        execution_mode=args.mode,
        rag_mode=args.rag_mode,
        judge=judge,
    )
    if not args.no_write:
        output_dir = args.output_dir or (
            pathlib.Path("data/eval/reports")
            / datetime.now().date().isoformat()
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        base = f"{args.suite}-{args.mode}"
        write_json_report(report, output_dir / f"{base}.json")
        write_markdown_report(report, path=output_dir / f"{base}.md")
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    return 0


def main() -> None:
    args = _parser().parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
