
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL = "/local/scratch/yicsun/models/NLLB-200-3.3B"

INPUT = Path(
    "eval/stage1_alignment/en-zh_source_sentences.jsonl"
)

OUTPUT = Path(
    "eval/stage1_alignment/en-zh_source_nllb_bridge_beam4.jsonl"
)

BATCH_SIZE = 16

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    src_lang="eng_Latn",
)

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
).cuda()
model.eval()

target_lang_id = tokenizer.convert_tokens_to_ids("zho_Hans")


# --------------------------------------------------
# Flatten source sentences while preserving structure
# --------------------------------------------------

items = []

with INPUT.open("r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)

        doc_id = obj["doc_id"]
        segment_id = obj["segment_id"]

        for sentence_id, sentence in enumerate(obj["sentences"]):
            items.append({
                "doc_id": doc_id,
                "segment_id": segment_id,
                "sentence_id": sentence_id,
                "source_sentence": sentence,
            })

print("Source sentences:", len(items))

if len(items) != 2463:
    raise RuntimeError(
        f"Expected 2463 source sentences, got {len(items)}"
    )


# --------------------------------------------------
# NLLB translation
# --------------------------------------------------

results = []

for start in range(0, len(items), BATCH_SIZE):
    batch_items = items[start:start + BATCH_SIZE]
    batch_texts = [x["source_sentence"] for x in batch_items]

    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=target_lang_id,
            max_new_tokens=256,
            num_beams=4,
        )

    translations = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
    )

    if len(translations) != len(batch_items):
        raise RuntimeError("Batch size mismatch.")

    for item, translation in zip(batch_items, translations):
        result = dict(item)
        result["nllb_bridge"] = translation.strip()
        results.append(result)

    print(
        f"\rTranslated {len(results)}/{len(items)}",
        end="",
        flush=True,
    )

print()


# --------------------------------------------------
# Save structured mapping
# --------------------------------------------------

with OUTPUT.open("w", encoding="utf-8") as f:
    for result in results:
        f.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )


# --------------------------------------------------
# Sanity checks
# --------------------------------------------------

missing = [
    x for x in results
    if not x["nllb_bridge"].strip()
]

print()
print("========== FINAL ==========")
print("source sentences:      ", len(items))
print("NLLB bridge sentences: ", len(results))
print("missing translations:  ", len(missing))
print("output:", OUTPUT)

print()
print("First 3:")
for x in results[:3]:
    print()
    print(
        f'doc={x["doc_id"]} '
        f'segment={x["segment_id"]} '
        f'sentence={x["sentence_id"]}'
    )
    print("SRC :", x["source_sentence"])
    print("NLLB:", x["nllb_bridge"])
