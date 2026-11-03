import json
import re
import subprocess
import tempfile
from pathlib import Path

REPO = Path("/local/scratch/yicsun/multiturn-llm-docmt_tine")
EVAL = REPO / "eval/ltcr"

FILES = {
    "Segment": EVAL / "sentence_alignment/segment/alignment_structured.jsonl",
    "Multi-turn": EVAL / "sentence_alignment/multiturn/alignment_structured.jsonl",
    "Source-primed": EVAL / "sentence_alignment/source_primed/alignment_structured.jsonl",
}

def parse_counts(text):
    exact = re.search(
        r"Exact:\s*(\d+)\s*/\s*(\d+)",
        text,
        flags=re.I,
    )
    fuzzy = re.search(
        r"Fuzzy:\s*(\d+)\s*/\s*(\d+)",
        text,
        flags=re.I,
    )

    if not exact or not fuzzy:
        return {}

    exact_match = int(exact.group(1))
    exact_total = int(exact.group(2))

    fuzzy_match = int(fuzzy.group(1))
    fuzzy_total = int(fuzzy.group(2))

    if exact_total != fuzzy_total:
        raise RuntimeError(
            f"Exact/Fuzzy denominators disagree: "
            f"{exact_total} vs {fuzzy_total}"
        )

    return {
        "exact": exact_match,
        "fuzzy": fuzzy_match,
        "total": exact_total,
    }


def evaluate(setting, jsonl):
    exact_sum = 0
    fuzzy_sum = 0
    total_sum = 0

    # Load sentence alignments and reconstruct document boundaries.
    from collections import defaultdict

    grouped = defaultdict(list)

    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                grouped[record["doc_id"]].append(record)

    expected_doc_ids = set(range(171))
    assert set(grouped) == expected_doc_ids, (
        f"{setting}: missing or unexpected document IDs: "
        f"{sorted(expected_doc_ids ^ set(grouped))}"
    )

    docs = []

    for doc_id in sorted(grouped):
        records = sorted(
            grouped[doc_id],
            key=lambda r: (
                r["article_id"],
                r["source_sentence_ids"][0],
                r["reference_sentence_ids"][0],
            ),
        )

        docs.append(records)

    print(f"\n===== {setting}: {len(docs)} documents =====")

    for doc_id, records in enumerate(docs):

        # Optional smoke test: evaluate only one document.
        import os
        test_doc = os.environ.get("LTCR_TEST_DOC")
        if test_doc is not None and doc_id != int(test_doc):
            continue

        src = [r["source"] for r in records]
        hyp = [r["reference"] for r in records]

        assert len(src) == len(hyp), (
            f"{setting} doc {doc_id}: "
            f"{len(src)} source vs {len(hyp)} hypothesis"
        )

        with tempfile.TemporaryDirectory(
            prefix=f"ltcr_{setting}_{doc_id}_"
        ) as tmp:

            tmp = Path(tmp)

            src_file = tmp / "src.txt"
            hyp_file = tmp / "hyp.txt"
            result_dir = tmp / "result"
            result_dir.mkdir()

            src_file.write_text(
                "\n".join(src) + "\n",
                encoding="utf-8"
            )
            hyp_file.write_text(
                "\n".join(hyp) + "\n",
                encoding="utf-8"
            )

            file_name = src_file.name

            token_file = result_dir / f"{file_name}.entity.token"
            tag_file = result_dir / f"{file_name}.entity.tag"
            align_file = result_dir / f"{file_name}.entity.align"
            word_file = result_dir / f"{file_name}.entity.word"
            prob_file = result_dir / f"{file_name}.entity.prob"
            pair_file = result_dir / f"{file_name}.entity_pair"
            out_file = result_dir / f"{file_name}.consistency"
            record_file = result_dir / f"{file_name}.record.json"

            subprocess.run([
                "python3", "-u",
                str(EVAL / "src_tgt_tokenize_en_to.py"),
                "-s", str(src_file),
                "-t", str(hyp_file),
                "-l", "en-zh",
                "-o", str(token_file),
                "--tag_file", str(tag_file),
                "--tool", "spacy",
                "--reverse",
            ], check=True)

            subprocess.run([
                "awesome-align",
                f"--data_file={token_file}",
                "--model_name_or_path=bert-base-multilingual-cased",
                "--extraction", "softmax",
                "--batch_size", "32",
                f"--output_file={align_file}",
                "--output_word_file", str(word_file),
                "--output_prob_file", str(prob_file),
            ], check=True)

            # Same blank-line removal as official run_eval.sh
            for p in [
                align_file,
                token_file,
                tag_file,
                word_file,
                prob_file,
            ]:
                lines = [
                    x for x in p.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if x.strip()
                ]
                p.write_text(
                    "\n".join(lines) + "\n",
                    encoding="utf-8"
                )

            cmd = [
                "python3", "-u",
                str(EVAL / "eval_probability_en_to.py"),
                "-a", str(align_file),
                "-t", str(token_file),
                "--tag_file", str(tag_file),
                "-p", "spacy",
                "--prob_file", str(prob_file),
                "-e", str(pair_file),
                "-o", str(out_file),
                "-r", str(record_file),
                "-l", "en-zh",
                "--stopwords", str(EVAL / "stopwords_en.txt"),
                "--reverse",
            ]

            r = subprocess.run(
                cmd,
                check=True,
                text=True,
                capture_output=True,
            )

            text = r.stdout + "\n" + r.stderr

            counts = parse_counts(text)

            if len(counts) != 3:
                print("\nCould not extract raw counts.")
                print("Setting:", setting)
                print("Document:", doc_id)
                print("Evaluator output:")
                print(text)
                raise RuntimeError(
                    "Need to inspect official evaluator output "
                    "before continuing."
                )

            exact_sum += counts["exact"]
            fuzzy_sum += counts["fuzzy"]
            total_sum += counts["total"]

        print(
            f"\r{setting}: {doc_id + 1}/{len(docs)}",
            end="",
            flush=True,
        )

    print()

    if total_sum == 0:
        ltcr1 = float("nan")
        ltcr1f = float("nan")
    else:
        ltcr1 = exact_sum / total_sum
        ltcr1f = fuzzy_sum / total_sum

    print(f"{setting}")
    print(f"exact_ent_match = {exact_sum}")
    print(f"fuzzy_ent_match = {fuzzy_sum}")
    print(f"ent_total = {total_sum}")
    print(f"LTCR-1  = {ltcr1:.6f}")
    print(f"LTCR-1f = {ltcr1f:.6f}")

    return {
        "setting": setting,
        "exact": exact_sum,
        "fuzzy": fuzzy_sum,
        "total": total_sum,
        "LTCR-1": ltcr1,
        "LTCR-1f": ltcr1f,
    }


results = []

for setting, path in FILES.items():
    results.append(evaluate(setting, path))

print("\n========== FINAL ==========")
for r in results:
    print(
        f'{r["setting"]:15s} '
        f'LTCR-1={r["LTCR-1"]:.6f}  '
        f'LTCR-1f={r["LTCR-1f"]:.6f}  '
        f'N={r["total"]}'
    )
