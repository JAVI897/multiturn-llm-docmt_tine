#!/usr/bin/env python3

"""
Character-level SLIDE evaluation using COMETKiwi.

This script is independent of the original segment-level SLIDE
implementation.

Method:
    1. Load complete source and translated documents.
    2. Construct sliding windows based on source character positions.
    3. Map source windows to target windows using proportional
       document positions.
    4. Evaluate each window with COMETKiwi.
    5. Average all window scores.

IMPORTANT:
    Proportional source-target mapping is approximate.
    It does not guarantee semantic alignment.

Example:

    python eval_slide_character.py data.jsonl \
        --window-size 200 \
        --stride 100 \
        --dry-run

    python eval_slide_character.py data.jsonl \
        --window-size 200 \
        --stride 100 \
        --batch-size 1 \
        --gpus 1 \
        --output-json results.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


# ============================================================
# 1. Load complete documents
# ============================================================

def load_documents_stag2(jsonl_path):
    """
    Load complete source and MT documents.

    Expected JSONL format:

    {
        "original": ["source segment 1", "source segment 2"],
        "translation": "complete translated document"
    }

    Also supports string-valued original and list-valued translation.

    Returns:
        List of (source_document, translated_document) tuples.
    """

    documents = []

    with open(jsonl_path, "r", encoding="utf-8") as f:

        for line_no, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_no}"
                ) from exc

            if not isinstance(obj, dict):
                raise ValueError(
                    f"Line {line_no}: expected a JSON object"
                )

            if "original" not in obj:
                raise ValueError(
                    f"Line {line_no}: missing 'original'"
                )

            if "translation" not in obj:
                raise ValueError(
                    f"Line {line_no}: missing 'translation'"
                )

            src = obj["original"]
            mt = obj["translation"]

            # Javi's original code:
            #
            # src = src.join(' ')
            #
            # Correct implementation:
            if isinstance(src, list):

                if not all(isinstance(x, str) for x in src):
                    raise ValueError(
                        f"Line {line_no}: original contains "
                        "non-string elements"
                    )

                src = " ".join(src)

            if isinstance(mt, list):

                if not all(isinstance(x, str) for x in mt):
                    raise ValueError(
                        f"Line {line_no}: translation contains "
                        "non-string elements"
                    )

                mt = " ".join(mt)

            if not isinstance(src, str):
                raise ValueError(
                    f"Line {line_no}: original must be "
                    "a string or list of strings"
                )

            if not isinstance(mt, str):
                raise ValueError(
                    f"Line {line_no}: translation must be "
                    "a string or list of strings"
                )

            src = src.strip()
            mt = mt.strip()

            if not src:
                raise ValueError(
                    f"Line {line_no}: empty source document"
                )

            if not mt:
                print(
                    f"[WARNING] Line {line_no}: "
                    "empty translation document, skipping."
                )
                continue

            documents.append((src, mt))

    return documents


# ============================================================
# 2. Construct character-level sliding windows
# ============================================================

def make_character_windows(
    documents,
    window_size=200,
    stride=100,
    include_remainder=False,
):
    """
    Construct character-level SLIDE windows.

    Window size:
        Number of source characters.

    Stride:
        Number of source characters to move forward.

    Overlap:
        window_size - stride

    Source-target correspondence:
        Approximate proportional mapping based on document lengths.

    Example:

        window_size = 200
        stride = 100

        Window 1: source[0:200]
        Window 2: source[100:300]
        Window 3: source[200:400]

    Windows never cross document boundaries.

    Returns:
        samples:
            List of COMETKiwi inputs.

        window_doc_ids:
            Document ID for each window.

        window_metadata:
            Character offsets and lengths.
    """

    if window_size <= 0:
        raise ValueError(
            "window_size must be greater than zero"
        )

    if stride <= 0:
        raise ValueError(
            "stride must be greater than zero"
        )

    if stride > window_size:
        raise ValueError(
            "stride cannot exceed window_size"
        )

    samples = []
    window_doc_ids = []
    window_metadata = []

    for doc_id, (src_doc, mt_doc) in enumerate(documents):

        src_len = len(src_doc)
        mt_len = len(mt_doc)

        if src_len == 0 or mt_len == 0:
            continue

        # Generate all complete windows.
        if src_len >= window_size:

            starts = list(
                range(
                    0,
                    src_len - window_size + 1,
                    stride,
                )
            )

        else:
            starts = []

        # Optionally include incomplete windows.
        #
        # For overlapping windows, append only the final
        # suffix that reaches the document end.
        if include_remainder:

            if not starts:
                starts = [0]

            elif starts[-1] + window_size < src_len:

                starts.append(
                    starts[-1] + stride
                )

        for window_index, start in enumerate(starts):

            end = min(
                start + window_size,
                src_len,
            )

            # Source character window.
            src_window = src_doc[start:end]

            # Proportional target character positions.
            #
            # IMPORTANT:
            # This is approximate, not sentence alignment.
            mt_start = round(
                (start / src_len) * mt_len
            )

            mt_end = round(
                (end / src_len) * mt_len
            )

            mt_start = max(
                0,
                min(mt_start, mt_len),
            )

            mt_end = max(
                mt_start,
                min(mt_end, mt_len),
            )

            mt_window = mt_doc[mt_start:mt_end]

            # Skip invalid empty windows.
            if not src_window.strip():
                continue

            if not mt_window.strip():
                continue

            samples.append({
                "src": src_window,
                "mt": mt_window,
            })

            window_doc_ids.append(doc_id)

            window_metadata.append({
                "doc_id": doc_id,
                "window_index": window_index,
                "src_start": start,
                "src_end": end,
                "mt_start": mt_start,
                "mt_end": mt_end,
                "src_characters": len(src_window),
                "mt_characters": len(mt_window),
            })

    return samples, window_doc_ids, window_metadata


# ============================================================
# 3. Print diagnostics
# ============================================================

def print_window_statistics(
    documents,
    samples,
    window_doc_ids,
    window_metadata,
    window_size,
    stride,
):
    """
    Print statistics about the constructed windows.
    """

    print()
    print("=" * 60)
    print("CHARACTER-LEVEL SLIDE WINDOW STATISTICS")
    print("=" * 60)

    print(f"Total documents: {len(documents)}")
    print(f"Total windows: {len(samples)}")

    print(f"Window size: {window_size} source characters")
    print(f"Stride: {stride} source characters")

    print(
        f"Overlap: {window_size - stride} "
        "source characters"
    )

    used_documents = len(set(window_doc_ids))

    print(f"Documents with windows: {used_documents}")

    print(
        "Documents without windows: "
        f"{len(documents) - used_documents}"
    )

    if not samples:
        print("No valid windows generated.")
        return

    src_lengths = np.array(
        [m["src_characters"] for m in window_metadata]
    )

    mt_lengths = np.array(
        [m["mt_characters"] for m in window_metadata]
    )

    print()
    print("Source window lengths:")

    print(
        f"  Mean: {np.mean(src_lengths):.2f}"
    )

    print(
        f"  Median: {np.median(src_lengths):.2f}"
    )

    print(
        f"  Min: {np.min(src_lengths)}"
    )

    print(
        f"  Max: {np.max(src_lengths)}"
    )

    print()
    print("Translation window lengths:")

    print(
        f"  Mean: {np.mean(mt_lengths):.2f}"
    )

    print(
        f"  Median: {np.median(mt_lengths):.2f}"
    )

    print(
        f"  Min: {np.min(mt_lengths)}"
    )

    print(
        f"  Max: {np.max(mt_lengths)}"
    )

    print("=" * 60)


# ============================================================
# 4. Preview windows
# ============================================================

def preview_windows(
    samples,
    window_metadata,
    num_preview=3,
):
    """
    Print several windows for manual inspection.
    """

    print()
    print("=" * 60)
    print("WINDOW PREVIEW")
    print("=" * 60)

    for i in range(
        min(num_preview, len(samples))
    ):

        sample = samples[i]
        metadata = window_metadata[i]

        print()
        print(f"Window {i + 1}")

        print(
            f"Document ID: {metadata['doc_id']}"
        )

        print(
            "Source offsets: "
            f"{metadata['src_start']}:"
            f"{metadata['src_end']}"
        )

        print(
            "Translation offsets: "
            f"{metadata['mt_start']}:"
            f"{metadata['mt_end']}"
        )

        print()
        print("SOURCE:")
        print(sample["src"])

        print()
        print("TRANSLATION:")
        print(sample["mt"])

        print("-" * 60)


# ============================================================
# 5. COMETKiwi evaluation
# ============================================================

def evaluate_with_comet(
    samples,
    model_name,
    batch_size,
    gpus,
):
    """
    Evaluate character windows using COMETKiwi.
    """

    # Import COMET only when evaluation is requested.
    # This allows --dry-run without loading the model.
    from comet import download_model, load_from_checkpoint

    print()
    print("=" * 60)
    print("LOADING COMET MODEL")
    print("=" * 60)

    print(f"Model: {model_name}")

    model_path = download_model(model_name)

    model = load_from_checkpoint(model_path)

    print()
    print("Starting COMETKiwi evaluation...")

    output = model.predict(
        samples,
        batch_size=batch_size,
        gpus=gpus,
    )

    scores = [
        float(score)
        for score in output.scores
    ]

    if len(scores) != len(samples):
        raise RuntimeError(
            "Number of COMET scores does not match "
            "number of windows"
        )

    if not np.all(np.isfinite(scores)):
        raise RuntimeError(
            "COMET returned non-finite scores"
        )

    return scores


# ============================================================
# 6. Save results
# ============================================================

def save_results(
    output_path,
    args,
    documents,
    samples,
    window_metadata,
    scores,
    slide_score,
):
    """
    Save evaluation configuration and window scores.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    window_results = []

    for sample, metadata, score in zip(
        samples,
        window_metadata,
        scores,
    ):

        record = {
            **metadata,
            "src": sample["src"],
            "mt": sample["mt"],
            "score": score,
        }

        window_results.append(record)

    results = {
        "metric": "SLIDE_Character_Proportional",
        "model": args.model,
        "input_file": str(args.jsonl),
        "window_size": args.window_size,
        "stride": args.stride,
        "overlap": args.window_size - args.stride,
        "include_remainder": args.include_remainder,
        "num_documents": len(documents),
        "num_windows": len(samples),
        "slide_score": slide_score,
        "windows": window_results,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"Results saved to: {output_path}")


# ============================================================
# 7. Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Character-level SLIDE evaluation "
            "using COMETKiwi"
        )
    )

    parser.add_argument(
        "jsonl",
        type=Path,
        help="Input generation JSONL file",
    )

    parser.add_argument(
        "--model",
        default="Unbabel/wmt22-cometkiwi-da",
        help="COMET model name",
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=200,
        help="Window size in source characters",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=100,
        help="Stride in source characters",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="COMET batch size",
    )

    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs for COMET",
    )

    parser.add_argument(
        "--include-remainder",
        action="store_true",
        help="Include incomplete final windows",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Construct and inspect windows without COMET",
    )

    parser.add_argument(
        "--preview",
        type=int,
        default=3,
        help="Number of windows to preview",
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for detailed JSON results",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate parameters
    # --------------------------------------------------------

    if not args.jsonl.is_file():
        parser.error(
            f"Input file does not exist: {args.jsonl}"
        )

    if args.window_size <= 0:
        parser.error(
            "--window-size must be positive"
        )

    if args.stride <= 0:
        parser.error(
            "--stride must be positive"
        )

    if args.stride > args.window_size:
        parser.error(
            "--stride cannot exceed --window-size"
        )

    if args.batch_size <= 0:
        parser.error(
            "--batch-size must be positive"
        )

    if args.gpus < 0:
        parser.error(
            "--gpus cannot be negative"
        )

    if args.preview < 0:
        parser.error(
            "--preview cannot be negative"
        )

    # --------------------------------------------------------
    # Load complete documents
    # --------------------------------------------------------

    print()
    print("Loading complete documents...")

    documents = load_documents_stag2(
        args.jsonl
    )

    if not documents:
        raise ValueError(
            "No documents found in input JSONL"
        )

    # --------------------------------------------------------
    # Construct character windows
    # --------------------------------------------------------

    print("Constructing character-level windows...")

    (
        samples,
        window_doc_ids,
        window_metadata,
    ) = make_character_windows(
        documents,
        window_size=args.window_size,
        stride=args.stride,
        include_remainder=args.include_remainder,
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print_window_statistics(
        documents,
        samples,
        window_doc_ids,
        window_metadata,
        args.window_size,
        args.stride,
    )

    preview_windows(
        samples,
        window_metadata,
        num_preview=args.preview,
    )

    # --------------------------------------------------------
    # Check for empty evaluation set
    # --------------------------------------------------------

    if not samples:
        raise ValueError(
            "No valid character windows generated. "
            "Try reducing --window-size or enabling "
            "--include-remainder."
        )

    # --------------------------------------------------------
    # Dry run
    # --------------------------------------------------------

    if args.dry_run:

        print()
        print("Dry run completed successfully.")
        print("COMETKiwi was not loaded.")

        return

    # --------------------------------------------------------
    # COMETKiwi evaluation
    # --------------------------------------------------------

    scores = evaluate_with_comet(
        samples,
        model_name=args.model,
        batch_size=args.batch_size,
        gpus=args.gpus,
    )

    # --------------------------------------------------------
    # Aggregate scores
    # --------------------------------------------------------

    slide_score = float(
        np.mean(scores)
    )

    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        f"Number of windows: {len(scores)}"
    )

    print(
        f"SLIDE (Character, proportional): "
        f"{slide_score:.6f}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    if args.output_json is not None:

        save_results(
            output_path=args.output_json,
            args=args,
            documents=documents,
            samples=samples,
            window_metadata=window_metadata,
            scores=scores,
            slide_score=slide_score,
        )


if __name__ == "__main__":
    main()