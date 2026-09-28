import argparse
import json

from sacrebleu import corpus_bleu


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def to_document_text(value):
    """
    Normalize one document into a single string.

    Salamandra / EuroLLM:
        translation/reference are already strings.

    MADLAD / Small100 Segment:
        translation/reference are lists of segments.
    """
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        return " ".join(str(x).strip() for x in value).strip()

    raise TypeError(
        f"Expected str or list, got {type(value).__name__}"
    )


def evaluate_dbleu(file_path, lang_direction):
    data = load_jsonl(file_path)

    if len(data) == 0:
        raise ValueError(f"Empty file: {file_path}")

    predictions = []
    references = []

    for i, item in enumerate(data):
        if "translation" not in item:
            raise KeyError(
                f"Row {i}: missing 'translation' in {file_path}"
            )

        if "reference" not in item:
            raise KeyError(
                f"Row {i}: missing 'reference' in {file_path}"
            )

        prediction = to_document_text(item["translation"])
        reference = to_document_text(item["reference"])

        predictions.append(prediction)
        references.append(reference)

    if len(predictions) != len(references):
        raise ValueError(
            f"Prediction/reference mismatch: "
            f"{len(predictions)} vs {len(references)}"
        )

    # IMPORTANT:
    # Do NOT remove the first row here.
    # Our 20-document manifest already excludes doc 0 (the canary).
    if lang_direction == "en-zh":
        score = corpus_bleu(
            predictions,
            [references],
            tokenize="zh",
        )
    else:
        score = corpus_bleu(
            predictions,
            [references],
        )

    return score


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        required=True,
        help="Evaluation JSONL output file",
    )

    parser.add_argument(
        "--lang_direction",
        required=True,
        choices=["en-zh", "en-es", "en-ca"],
    )

    args = parser.parse_args()

    data = load_jsonl(args.input_file)

    print("Input file:", args.input_file)
    print("Number of documents:", len(data))

    score = evaluate_dbleu(
        args.input_file,
        args.lang_direction,
    )

    print("dBLEU:", round(score.score, 4))
    print("SacreBLEU:", score)


if __name__ == "__main__":
    main()