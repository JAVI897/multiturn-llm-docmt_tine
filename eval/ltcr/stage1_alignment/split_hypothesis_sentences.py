import json
import re
from pathlib import Path

input_path = Path(
    "outputs/EuroLLM-22B-Instruct-2512/"
    "eurollm2512_en-zh_singleturn_gen.jsonl"
)

output_path = Path(
    "eval/stage1_alignment/"
    "en-zh_hypothesis_sentences.jsonl"
)

def split_chinese_sentences(text):
    pattern = r'[。！？!?]+[”’」』）】]*|(?<!\.)\.{3,}(?!\.)|…{1,}|\n+|$'

    sentences = []
    start = 0

    for match in re.finditer(pattern, text):
        end = match.end()
        sentence = text[start:end].strip()

        if sentence:
            sentences.append(sentence)

        start = end

    return sentences


def remove_whitespace(text):
    return re.sub(r'\s+', '', text)


total_sentences = 0
valid_documents = 0

with input_path.open(encoding="utf-8") as fin, \
     output_path.open("w", encoding="utf-8") as fout:

    for doc_id, line in enumerate(fin):
        record = json.loads(line)
        translation = record["translation"]

        if translation.strip().startswith("CANARY GUID"):
            continue

        sentences = split_chinese_sentences(translation)

        reconstructed = "".join(sentences)

        assert remove_whitespace(reconstructed) == remove_whitespace(translation), (
            f"Content mismatch in doc_id={doc_id}"
        )

        result = {
            "doc_id": doc_id,
            "sentences": sentences
        }

        fout.write(json.dumps(result, ensure_ascii=False) + "\n")

        valid_documents += 1
        total_sentences += len(sentences)

print("Documents:", valid_documents)
print("Total sentences:", total_sentences)
print("Output:", output_path)