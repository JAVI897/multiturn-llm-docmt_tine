#!/usr/bin/env python3
"""Evaluate WMT24 English-to-Chinese translations with sentence-aligned LTCR."""

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


EVAL_DIR = Path(__file__).resolve().parent
REPO_DIR = EVAL_DIR.parents[1]

MODES = {
    "single": "single_turn",
    "segment": "segment",
    "mturn": "multi_turn",
    "sourceprime": "source_primed",
}


def load_alignment(args):
    """Reuse the existing WMT24 alignment implementation."""
    path = EVAL_DIR / "sentence_alignment/run_sentence_alignment.py"
    spec = importlib.util.spec_from_file_location("ltcr_alignment", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load alignment module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # An absolute input path overrides the legacy model-specific prefix.
    module.INPUTS = {
        MODES[mode]: str(input_path)
        for mode, input_path in args.inputs.items()
    }
    module.BASE = REPO_DIR
    module.HIST = args.stage1_dir
    module.PYTHON = sys.executable
    module.BLEUALIGN = str(args.bleualign)
    return module


def replace_once(source, old, new):
    """Fail explicitly if the legacy evaluator no longer matches."""
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected one evaluator compatibility pattern, found {count}: {old}"
        )
    return source.replace(old, new, 1)


def load_evaluator(batch_size):
    """Load evaluator functions without executing its historical batch driver.

    These explicit compatibility adaptations preserve the existing LTCR
    calculation while accepting the composed source-hypothesis schema.
    """
    path = EVAL_DIR / "eval_sentence_docaware.py"
    source = path.read_text(encoding="utf-8")

    replacements = [
        (
            'r["source_sentence_ids"][0]',
            'r["source_sentence_keys"][0]["sentence_id"]',
        ),
        (
            'hyp = [r["reference"] for r in records]',
            'hyp = [r["hypothesis"] for r in records]',
        ),
        (
            "assert set(grouped) == expected_doc_ids, (",
            "assert set(grouped) <= expected_doc_ids, (",
        ),
        (
            "docs.append(records)",
            "docs.append((doc_id, records))",
        ),
        (
            "for doc_id, records in enumerate(docs):",
            "for doc_id, records in docs:",
        ),
        (
            '"--batch_size", "32"',
            f'"--batch_size", "{batch_size}"',
        ),
        (
            '"python3", "-u"',
            "sys.executable, \"-u\"",
        ),
    ]

    # The interpreter appears in both subprocess calls.
    interpreter_old, interpreter_new = replacements.pop()
    if source.count(interpreter_old) != 2:
        raise RuntimeError("Unexpected evaluator subprocess configuration.")
    source = source.replace(interpreter_old, interpreter_new)

    for old, new in replacements:
        source = replace_once(source, old, new)

    tree = ast.parse(source, filename=str(path))
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
    ]
    namespace = {
        "__name__": "ltcr_evaluator",
        "__file__": str(path),
        "sys": sys,
        "REPO": REPO_DIR,
        "EVAL": EVAL_DIR,
    }
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["evaluate"]


def save_results(path, results):
    """Save completed configurations through an atomic file replacement."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Model label for reporting; no predefined model restriction.",
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="MODE=JSONL",
        help=(
            "Repeat for multiple modes. Modes: "
            "single, segment, mturn, sourceprime."
        ),
    )
    parser.add_argument(
        "--stage1-dir",
        type=Path,
        default=EVAL_DIR / "stage1_alignment",
        help="Directory containing the archived WMT24 Stage 1 files.",
    )
    parser.add_argument(
        "--bleualign",
        type=Path,
        required=True,
        help="Path to bleualign.py in the patched Bleualign checkout.",
    )
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--align-batch-size", type=int, default=8)
    parser.add_argument("--test-doc", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.processes < 1 or args.align_batch_size < 1:
        parser.error("Process count and alignment batch size must be positive.")
    if args.test_doc is not None and not 0 <= args.test_doc < 171:
        parser.error("--test-doc must be between 0 and 170.")

    args.inputs = {}
    for item in args.input:
        mode, separator, filename = item.partition("=")
        if not separator or mode not in MODES or not filename:
            parser.error(f"Invalid input specification: {item}")
        if mode in args.inputs:
            parser.error(f"Duplicate mode: {mode}")
        path = Path(filename).expanduser().resolve()
        if not path.is_file():
            parser.error(f"Input file not found: {path}")
        args.inputs[mode] = path

    args.stage1_dir = args.stage1_dir.expanduser().resolve()
    args.bleualign = args.bleualign.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    if not args.bleualign.is_file():
        parser.error(f"Bleualign entry point not found: {args.bleualign}")

    for filename in [
        "en-zh_source_sentences.jsonl",
        "en-zh_reference_sentences.jsonl",
        "stage1_alignment_structured.jsonl",
    ]:
        if not (args.stage1_dir / filename).is_file():
            parser.error(f"Missing Stage 1 file: {args.stage1_dir / filename}")

    if args.output.exists():
        parser.error("Output already exists; choose a new filename.")

    return args


def main():
    args = parse_args()
    alignment = load_alignment(args)
    evaluate = load_evaluator(args.align_batch_size)
    results = []

    previous_test_doc = os.environ.get("LTCR_TEST_DOC")
    if args.test_doc is None:
        os.environ.pop("LTCR_TEST_DOC", None)
    else:
        os.environ["LTCR_TEST_DOC"] = str(args.test_doc)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        for mode, input_path in args.inputs.items():
            setting = MODES[mode]
            print(f"Preparing: {args.model} | {mode}", flush=True)

            with tempfile.TemporaryDirectory(prefix="ltcr_en_zh_") as temporary:
                work = Path(temporary) / "alignment"
                articles, stage1, source_lookup = alignment.prepare(setting, work)

                try:
                    stage2 = alignment.run_bleualign(work, args.processes)
                except Exception:
                    for filename in ["bleualign_command.json", "bleualign.log"]:
                        path = work / filename
                        if path.exists():
                            print(f"\n===== {filename} =====", flush=True)
                            print(
                                path.read_text(errors="replace")[-20000:],
                                flush=True,
                            )
                    raise

                statistics = alignment.compose(
                    setting, work, articles, stage1, source_lookup, stage2
                )
                aligned_path = work / "source_hypothesis_alignment.jsonl"
                rows = alignment.read_jsonl(aligned_path)
                empty = alignment.read_jsonl(work / "empty_hypotheses.jsonl")

                if args.test_doc is not None:
                    rows = [
                        row for row in rows if row["doc_id"] == args.test_doc
                    ]
                    alignment.write_jsonl(aligned_path, rows)

                if not rows:
                    raise RuntimeError("No usable alignment groups to evaluate.")

                present = {row["doc_id"] for row in rows}
                expected = (
                    {args.test_doc}
                    if args.test_doc is not None
                    else set(range(171))
                )
                statistics["documents_evaluated"] = sorted(present)
                statistics["documents_without_usable_alignment"] = sorted(
                    expected - present
                )

                result = evaluate(f"{args.model}__{mode}", aligned_path)
                if result["total"] == 0:
                    raise RuntimeError(
                        "No eligible entity comparisons; LTCR is undefined."
                    )

                result.update(
                    model=args.model,
                    mode=mode,
                    input=str(input_path),
                    test_doc=args.test_doc,
                    align_batch_size=args.align_batch_size,
                    alignment_stats=statistics,
                    empty_hypothesis_articles=len(empty),
                )
                results.append(result)
                save_results(args.output, results)

            print(f"Completed: {args.model} | {mode}", flush=True)

    finally:
        if previous_test_doc is None:
            os.environ.pop("LTCR_TEST_DOC", None)
        else:
            os.environ["LTCR_TEST_DOC"] = previous_test_doc

    print(f"Results saved to: {args.output}", flush=True)


if __name__ == "__main__":
    main()
