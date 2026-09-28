import argparse
import json
import os
import sys

# Amazon Doc-COMET fork
COMET_DIR = "/local/scratch/yicsun/doc-mt-metrics/COMET"
sys.path.insert(0, COMET_DIR)

from comet import load_from_checkpoint
from add_context import add_context


DEFAULT_CKPT = (
    "/local/scratch/yicsun/models/"
    "wmt21-comet-mqm/checkpoints/model.ckpt"
)


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at line {line_num}: {e}"
                )

    return rows


def flatten_documents(rows):
    src_all = []
    mt_all = []
    ref_all = []
    doc_ids = []

    for doc_idx, item in enumerate(rows):

        if "original" not in item:
            raise KeyError(
                f"Document {doc_idx}: missing 'original'"
            )

        if "translation_split" not in item:
            raise KeyError(
                f"Document {doc_idx}: missing 'translation_split'"
            )

        if "reference_split" not in item:
            raise KeyError(
                f"Document {doc_idx}: missing 'reference_split'"
            )

        src = item["original"]
        mt = item["translation_split"]
        ref = item["reference_split"]

        if not (
            isinstance(src, list)
            and isinstance(mt, list)
            and isinstance(ref, list)
        ):
            raise TypeError(
                f"Document {doc_idx}: "
                "original / translation_split / reference_split "
                "must all be lists."
            )

        if not (len(src) == len(mt) == len(ref)):
            raise ValueError(
                f"Document {doc_idx}: segment count mismatch: "
                f"src={len(src)}, mt={len(mt)}, ref={len(ref)}"
            )

        for s, m, r in zip(src, mt, ref):
            src_all.append(str(s).strip())
            mt_all.append(str(m).strip())
            ref_all.append(str(r).strip())
            doc_ids.append(f"doc_{doc_idx}")

    return src_all, mt_all, ref_all, doc_ids


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Translation JSONL file",
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CKPT,
        help="Path to wmt21-comet-mqm checkpoint",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="Number of GPUs used by COMET. 0 = CPU.",
    )

    args = parser.parse_args()

    print("Input:", args.input)
    print("Checkpoint:", args.checkpoint)

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    # --------------------------------------------------
    # 1. Read documents
    # --------------------------------------------------

    rows = load_jsonl(args.input)

    print("Documents:", len(rows))

    # --------------------------------------------------
    # 2. Flatten documents while preserving doc IDs
    # --------------------------------------------------

    src, mt, ref, doc_ids = flatten_documents(rows)

    print("Segments:", len(src))

    if len(src) == 0:
        raise ValueError("No segments found.")

    # --------------------------------------------------
    # 3. Load official wmt21-comet-mqm checkpoint
    # --------------------------------------------------

    print("\nLoading COMET model...")

    model = load_from_checkpoint(args.checkpoint)

    if not hasattr(model, "set_document_level"):
        raise RuntimeError(
            "Loaded COMET model does not support "
            "set_document_level()."
        )

    model.set_document_level()

    print("Document-level mode: ON")

    sep_token = model.encoder.tokenizer.sep_token

    print("SEP token:", repr(sep_token))

    # --------------------------------------------------
    # 4. Official Doc-COMET context construction
    #
    # add_context default:
    #     ws = 2
    #
    # SRC:
    #     previous source + current source
    #
    # REF:
    #     previous reference + current reference
    #
    # MT:
    #     previous REFERENCE + current MT
    #
    # This follows reference-based Doc-COMET.
    # --------------------------------------------------

    print("\nBuilding document context...")

    src_ctx = add_context(
        orig_txt=src,
        context=src,
        doc_ids=doc_ids,
        sep_token=sep_token,
    )

    ref_ctx = add_context(
        orig_txt=ref,
        context=ref,
        doc_ids=doc_ids,
        sep_token=sep_token,
    )

    mt_ctx = add_context(
        orig_txt=mt,
        context=ref,
        doc_ids=doc_ids,
        sep_token=sep_token,
    )

    assert len(src_ctx) == len(mt_ctx) == len(ref_ctx)

    # --------------------------------------------------
    # 5. Show first few examples for sanity checking
    # --------------------------------------------------

    print("\nFirst context examples:")

    for i in range(min(3, len(src_ctx))):
        print(f"\n--- Segment {i + 1} ---")
        print("DOC:", doc_ids[i])
        print("SRC:", src_ctx[i])
        print("MT :", mt_ctx[i])
        print("REF:", ref_ctx[i])

    # --------------------------------------------------
    # 6. Build COMET input
    # --------------------------------------------------

    data = [
        {
            "src": s,
            "mt": m,
            "ref": r,
        }
        for s, m, r in zip(
            src_ctx,
            mt_ctx,
            ref_ctx,
        )
    ]

    # --------------------------------------------------
    # 7. Predict
    # --------------------------------------------------

    print("\nRunning Doc-COMET...")
    print("Batch size:", args.batch_size)
    print("GPUs:", args.gpus)

    segment_scores, system_score = model.predict(
        data,
        batch_size=args.batch_size,
        gpus=args.gpus,
    )

    # --------------------------------------------------
    # 8. Results
    # --------------------------------------------------

    print("\n========================================")
    print("Doc-COMET evaluation")
    print("========================================")
    print("Documents:", len(rows))
    print("Segments:", len(segment_scores))
    print("Doc-COMET:", float(system_score))
    print("========================================")


if __name__ == "__main__":
    main()
