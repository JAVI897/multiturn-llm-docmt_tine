import json
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent


def load_jsonl(filename):
    path = BASE / filename

    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# Read final alignment file from disk
alignments = load_jsonl("source_hypothesis_alignment.jsonl")

print("Total alignment records:", len(alignments))

assert len(alignments) == 2099


# Check document IDs and group IDs
group_ids_by_doc = defaultdict(list)

for record in alignments:

    doc_id = record["doc_id"]
    group_id = record["group_id"]

    assert 1 <= doc_id <= 170
    assert isinstance(group_id, int)

    group_ids_by_doc[doc_id].append(group_id)


print("Documents represented:", len(group_ids_by_doc))

assert len(group_ids_by_doc) == 170


# Check group IDs are consecutive within each document
for doc_id, group_ids in group_ids_by_doc.items():

    expected = list(range(len(group_ids)))

    assert group_ids == expected, (
        f"Invalid group IDs in document {doc_id}"
    )


print("PASS: Document IDs and group IDs validated!")


# --------------------------------------------------
# Validate Source and Hypothesis text
# --------------------------------------------------

source_docs = load_jsonl("en-zh_source_sentences.jsonl")
hypothesis_docs = load_jsonl("en-zh_hypothesis_sentences.jsonl")


# Build original Source sentence lookup
source_lookup = {}

for record in source_docs:

    doc_id = record["doc_id"]
    segment_id = record["segment_id"]

    if doc_id == 0:
        continue

    for sentence_id, sentence in enumerate(record["sentences"]):

        key = (doc_id, segment_id, sentence_id)

        assert key not in source_lookup
        source_lookup[key] = sentence


# Build original Hypothesis sentence lookup
hypothesis_lookup = {}

for record in hypothesis_docs:

    doc_id = record["doc_id"]

    for sentence_id, sentence in enumerate(record["sentences"]):

        key = (doc_id, sentence_id)

        assert key not in hypothesis_lookup
        hypothesis_lookup[key] = sentence


# Validate every alignment record
for record in alignments:

    doc_id = record["doc_id"]

    source_sentences = []

    for item in record["source_sentence_ids"]:

        key = (
            doc_id,
            item["segment_id"],
            item["sentence_id"]
        )

        assert key in source_lookup
        source_sentences.append(source_lookup[key])

    hypothesis_sentences = []

    for sentence_id in record["hypothesis_sentence_ids"]:

        key = (doc_id, sentence_id)

        assert key in hypothesis_lookup
        hypothesis_sentences.append(hypothesis_lookup[key])

    expected_source = " ".join(source_sentences)
    expected_hypothesis = " ".join(hypothesis_sentences)

    assert record["source"] == expected_source, (
        f"Source text mismatch: doc={doc_id}, "
        f"group={record['group_id']}"
    )

    assert record["hypothesis"] == expected_hypothesis, (
        f"Hypothesis text mismatch: doc={doc_id}, "
        f"group={record['group_id']}"
    )


print("Source sentences available:", len(source_lookup))
print("Hypothesis sentences available:", len(hypothesis_lookup))
print("Alignment records validated:", len(alignments))

print("PASS: All Source and Hypothesis texts match original data!")

# --------------------------------------------------
# Validate sentence ordering, uniqueness and coverage
# --------------------------------------------------

seen_source = set()
seen_hypothesis = set()

for record in alignments:

    doc_id = record["doc_id"]

    source_ids = [
        (item["segment_id"], item["sentence_id"])
        for item in record["source_sentence_ids"]
    ]

    hypothesis_ids = record["hypothesis_sentence_ids"]

    # Check sentence order within each group
    assert source_ids == sorted(source_ids)
    assert hypothesis_ids == sorted(hypothesis_ids)

    # Every group must contain both Source and Hypothesis
    assert len(source_ids) > 0
    assert len(hypothesis_ids) > 0

    # Check Source sentence uniqueness
    for segment_id, sentence_id in source_ids:

        key = (doc_id, segment_id, sentence_id)

        assert key not in seen_source, (
            f"Duplicate Source sentence: {key}"
        )

        seen_source.add(key)

    # Check Hypothesis sentence uniqueness
    for sentence_id in hypothesis_ids:

        key = (doc_id, sentence_id)

        assert key not in seen_hypothesis, (
            f"Duplicate Hypothesis sentence: {key}"
        )

        seen_hypothesis.add(key)


# Check expected sentence coverage
assert len(seen_source) == 2300
assert len(seen_hypothesis) == 2178

source_coverage = len(seen_source) / len(source_lookup)
hypothesis_coverage = len(seen_hypothesis) / len(hypothesis_lookup)

print("Unique Source sentences:", len(seen_source))
print("Unique Hypothesis sentences:", len(seen_hypothesis))

print(f"Source coverage: {source_coverage:.2%}")
print(f"Hypothesis coverage: {hypothesis_coverage:.2%}")

print("PASS: Final alignment integrity validated!")