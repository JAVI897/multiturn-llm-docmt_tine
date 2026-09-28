import argparse
import json


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def extract_segments(item):
    """
    Support two schemas:

    Salamandra / EuroLLM:
        original
        translation_split
        reference_split

    MADLAD / Small100:
        source
        translation
        reference
    """

    if "translation_split" in item:
        src = item["original"]
        mt = item["translation_split"]
        ref = item["reference_split"]
    else:
        src = item["source"]
        mt = item["translation"]
        ref = item["reference"]

    if not isinstance(src, list):
        raise TypeError("COMET requires segment-aligned source lists.")

    if not isinstance(mt, list):
        raise TypeError("COMET requires segment-aligned translation lists.")

    if not isinstance(ref, list):
        raise TypeError("COMET requires segment-aligned reference lists.")

    if not (len(src) == len(mt) == len(ref)):
        raise ValueError(
            f"Segment mismatch: "
            f"source={len(src)}, translation={len(mt)}, reference={len(ref)}"
        )

    return src, mt, ref


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    # Import here so simple syntax/file checks don't need COMET loaded.
    from comet import download_model, load_from_checkpoint

    rows = load_jsonl(args.input_file)

    comet_data = []

    for item in rows:
        src, mt, ref = extract_segments(item)

        for s, m, r in zip(src, mt, ref):
            comet_data.append({
                "src": str(s).strip(),
                "mt": str(m).strip(),
                "ref": str(r).strip(),
            })

    print("Input file:", args.input_file)
    print("Number of documents:", len(rows))
    print("Number of segments:", len(comet_data))

    # IMPORTANT:
    # Our fixed 20-document subset already excludes doc0 canary.
    # Therefore DO NOT use [1:] here.

    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)

    output = model.predict(
        comet_data,
        batch_size=8,
        gpus=args.gpus,
    )

    print("COMET:", round(float(output.system_score), 4))


if __name__ == "__main__":
    main()
