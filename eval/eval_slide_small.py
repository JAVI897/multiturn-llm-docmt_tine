
#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
from comet import download_model, load_from_checkpoint


def load_documents(jsonl_path):
    """Load source/MT documents from TINE generation JSONL."""
    documents = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            if "original" not in obj:
                raise KeyError(f"Line {line_no}: missing 'original'")
            if "translation_split" not in obj:
                raise KeyError(f"Line {line_no}: missing 'translation_split'")

            src = obj["original"]
            mt = obj["translation_split"]

            # TINE files normally store original as a list of segments.
            # Handle a string defensively as well.
            if isinstance(src, str):
                src = [src]
            if isinstance(mt, str):
                mt = [mt]

            if len(src) != len(mt):
                raise ValueError(
                    f"Line {line_no}: source/MT length mismatch: "
                    f"{len(src)} vs {len(mt)}"
                )

            documents.append((src, mt))

    return documents


def make_slide_windows(documents, window_size=6, stride=6):
    """
    Construct non-overlapping SLIDE windows.

    Only complete windows are retained. Therefore:
      - documents shorter than window_size are skipped;
      - trailing remainder shorter than window_size is skipped.
    """
    samples = []
    window_doc_ids = []

    for doc_id, (src_doc, mt_doc) in enumerate(documents):
        n = len(src_doc)

        for start in range(0, n - window_size + 1, stride):
            end = start + window_size

            src_window = " ".join(s.strip() for s in src_doc[start:end])
            mt_window = " ".join(s.strip() for s in mt_doc[start:end])

            samples.append({
                "src": src_window,
                "mt": mt_window,
            })
            window_doc_ids.append(doc_id)

    return samples, window_doc_ids


def main():
    parser = argparse.ArgumentParser(
        description="SLIDE-small document-level MT evaluation"
    )
    parser.add_argument("jsonl", help="TINE generated JSONL file")
    parser.add_argument(
        "--model",
        default="Unbabel/wmt22-cometkiwi-da",
        help="COMETKiwi checkpoint",
    )
    parser.add_argument("--window-size", type=int, default=6)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=1)

    args = parser.parse_args()

    path = Path(args.jsonl)
    if not path.exists():
        raise FileNotFoundError(path)

    documents = load_documents(path)

    samples, window_doc_ids = make_slide_windows(
        documents,
        window_size=args.window_size,
        stride=args.stride,
    )

    if not samples:
        raise RuntimeError("No complete SLIDE windows were produced.")

    docs_used = len(set(window_doc_ids))
    docs_skipped = len(documents) - docs_used

    print(f"Input: {path}")
    print(f"Documents: {len(documents)}")
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}")
    print(f"Valid windows: {len(samples)}")
    print(f"Documents contributing >=1 window: {docs_used}")
    print(f"Documents skipped (< full window): {docs_skipped}")
    print(f"Model: {args.model}")

    print("\nLoading COMETKiwi...")
    model_path = download_model(args.model)
    model = load_from_checkpoint(model_path)

    print("Running SLIDE-small evaluation...")
    output = model.predict(
        samples,
        batch_size=args.batch_size,
        gpus=args.gpus,
    )

    scores = [float(x) for x in output.scores]

    if len(scores) != len(samples):
        raise RuntimeError(
            f"Expected {len(samples)} scores, got {len(scores)}"
        )

    slide_score = float(np.mean(scores))

    print("\n========== SLIDE-small ==========")
    print(f"Windows scored: {len(scores)}")
    print(f"Mean window score: {slide_score:.6f}")
    print(f"Min window score: {min(scores):.6f}")
    print(f"Max window score: {max(scores):.6f}")
    print("=================================")


if __name__ == "__main__":
    main()


