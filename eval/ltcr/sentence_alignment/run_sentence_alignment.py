#!/usr/bin/env python3
"""Llama en-zh Stage2 Reference<->Hypothesis alignment and Stage1 composition.

Run on habra. Requires historical Stage1 files and locally modified Bleualign.
"""
import argparse
import collections
import datetime
import json
import os
import pathlib
import re
import subprocess
import sys

BASE = pathlib.Path('/local/scratch/yicsun/multiturn-llm-docmt_tine')
HIST = pathlib.Path('/local/scratch/yicsun/multiturn-llm-docmt_tine/eval/ltcr/stage1_alignment')
PYTHON = '/local/scratch/yicsun/ltcr_env/bin/python'
BLEUALIGN = '/local/scratch/yicsun/Bleualign/bleualign.py'
INPUTS = {}
SPLIT = re.compile(r'[。！？!?]+[”’」』）】]*|(?<!\.)\.{3,}(?!\.)|…{1,}|\n+|$')


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def split_zh(text):
    assert isinstance(text, str), type(text)
    out = []
    start = 0
    for match in SPLIT.finditer(text):
        sentence = text[start:match.end()].strip()
        if sentence:
            out.append(sentence)
        start = match.end()
    return out


def normalize(text):
    return ' '.join(text.split())


def emit_articles(path, articles):
    with open(path, 'w', encoding='utf-8') as f:
        for sentences in articles:
            for sentence in sentences:
                if '\n' in sentence or '\r' in sentence or sentence.strip() != sentence:
                    raise ValueError('Invalid Bleualign sentence: ' + repr(sentence))
                f.write(sentence + '\n')
            f.write('.EOA\n')


def fail(msg):
    raise RuntimeError(msg)


def prepare(setting, output):
    inputs = pathlib.Path(INPUTS[setting])
    data = read_jsonl(inputs)
    if len(data) != 171:
        fail(f'{setting}: expected 171 docs, got {len(data)}')
    ref_rows = read_jsonl(HIST / 'en-zh_reference_sentences.jsonl')
    src_rows = read_jsonl(HIST / 'en-zh_source_sentences.jsonl')
    if len(ref_rows) != 998 or len(src_rows) != 998:
        fail('Historical sentence splits do not have 998 segments')
    ref_by_key = {}
    src_by_key = {}
    for r in ref_rows:
        key = (int(r['doc_id']), int(r['segment_id']))
        if key in ref_by_key: fail('Duplicate ref key: ' + str(key))
        ref_by_key[key] = r['sentences']
    for r in src_rows:
        key = (int(r['doc_id']), int(r['segment_id']))
        if key in src_by_key: fail('Duplicate src key: ' + str(key))
        src_by_key[key] = r['sentences']
    if set(ref_by_key) != set(src_by_key): fail('Source/reference segment keys differ')
    if set(d for d, _ in ref_by_key) != set(range(171)):
        fail('Historical ref doc ids are not 0..170')
    stage1 = read_jsonl(HIST / 'stage1_alignment_structured.jsonl')
    for row in stage1:
        key = (int(row['doc_id']), int(row['segment_id']))
        if key not in ref_by_key: fail('Stage1 unknown segment: ' + str(key))
        for idx in row['reference_sentence_ids']:
            if not 0 <= idx < len(ref_by_key[key]): fail('Stage1 reference ID out of range')
        for idx in row['source_sentence_ids']:
            if not 0 <= idx < len(src_by_key[key]): fail('Stage1 source ID out of range')
    empty_hypotheses = []
    if setting == 'single_turn':
        ref_docs = collections.defaultdict(list)
        ref_index = collections.defaultdict(list)
        for key in sorted(ref_by_key):
            doc, seg = key
            for local_id, sentence in enumerate(ref_by_key[key]):
                ref_docs[doc].append(sentence)
                ref_index[doc].append((seg, local_id))
        articles = []
        for doc, row in enumerate(data):
            if not isinstance(row['original'], str) or not isinstance(row['translation'], str):
                fail(f'Single-turn doc {doc} has wrong types')
            original_segments = [src_rows_i['original_segment'] for src_rows_i in src_rows if int(src_rows_i['doc_id']) == doc]
            if normalize(row['original']) != normalize(' '.join(original_segments)):
                fail(f'Single-turn original source mismatch at doc {doc}')
            ref_sents = ref_docs[doc]
            hyp_sents = split_zh(row['translation'])
            if not hyp_sents:
                empty_hypotheses.append({
                    'setting': setting,
                    'doc_id': doc,
                    'segment_id': None,
                    'raw_translation': row['translation'],
                })
                continue
            articles.append({'doc_id': doc, 'segment_id': None, 'sentences': ref_sents,
                             'reference_local_keys': [{'segment_id': seg, 'sentence_id': sid} for seg, sid in ref_index[doc]],
                             'hypothesis_sentences': hyp_sents})
    else:
        articles = []
        for doc, row in enumerate(data):
            orig = row['original']
            hyp = row['translation_split']
            if not isinstance(orig, list) or not isinstance(hyp, list) or len(orig) != len(hyp):
                fail(f'{setting}: invalid segment arrays at doc {doc}')
            expected = sorted(seg for d, seg in ref_by_key if d == doc)
            if expected != list(range(len(orig))):
                fail(f'{setting}: segment count mismatch at doc {doc}: {len(orig)} vs {len(expected)}')
            for seg, (original, translation) in enumerate(zip(orig, hyp)):
                key = (doc, seg)
                src_original = next(r['original_segment'] for r in src_rows if int(r['doc_id']) == doc and int(r['segment_id']) == seg)
                if normalize(original) != normalize(src_original):
                    fail(f'{setting}: original source mismatch at {key}')
                hyp_sents = split_zh(translation)
                if not hyp_sents:
                    empty_hypotheses.append({
                        'setting': setting,
                        'doc_id': doc,
                        'segment_id': seg,
                        'raw_translation': translation,
                    })
                    continue
                articles.append({'doc_id': doc, 'segment_id': seg, 'sentences': ref_by_key[key],
                                 'reference_local_keys': [{'segment_id': seg, 'sentence_id': sid} for sid in range(len(ref_by_key[key]))],
                                 'hypothesis_sentences': hyp_sents})
    expected = 171 if setting == 'single_turn' else 998
    if len(articles) + len(empty_hypotheses) != expected:
        fail(f'{setting}: article accounting mismatch')
    if not articles:
        fail(f'{setting}: all hypotheses are empty')
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / 'empty_hypotheses.jsonl', empty_hypotheses)
    write_jsonl(output / 'article_metadata.jsonl', articles)
    emit_articles(output / 'reference.txt', [a['sentences'] for a in articles])
    emit_articles(output / 'hypothesis.txt', [a['hypothesis_sentences'] for a in articles])
    return articles, stage1, src_by_key


def run_bleualign(output, processes):
    export = output / 'stage2_alignment_structured.jsonl'
    env = dict(os.environ, BLEUALIGN_EXPORT_STAGE='stage2',
               BLEUALIGN_EXPORT_JSONL=str(export),
               BLEUALIGN_ARTICLE_METADATA_JSONL=str(output / 'article_metadata.jsonl'))
    command = [PYTHON, BLEUALIGN, '--source', str(output / 'reference.txt'),
               '--target', str(output / 'hypothesis.txt'), '--output', str(output / 'stage2_aligned'),
               '--srctotarget', '-', '--bleu_charlevel', '--bleu_n', '2',
               '--processes', str(processes)]
    (output / 'bleualign_command.json').write_text(json.dumps(command, indent=2) + '\n', encoding='utf-8')
    with open(output / 'bleualign.log', 'w', encoding='utf-8') as log:
        proc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if proc.returncode:
        fail(f'Bleualign failed (exit {proc.returncode}); inspect {output / "bleualign.log"}')
    if not export.exists(): fail('Bleualign did not export stage2 JSONL')
    return read_jsonl(export)


def compose(setting, output, articles, stage1, src_by_key, stage2):
    source_lookup = collections.defaultdict(set)
    for row in stage1:
        key = (int(row['doc_id']), int(row['segment_id']))
        for ref_id in row['reference_sentence_ids']:
            source_lookup[(key[0], key[1], int(ref_id))].update(int(x) for x in row['source_sentence_ids'])
    result = []
    stats = collections.Counter()
    seen_articles = set()
    for row in stage2:
        article_id = int(row['article_id'])
        if not 0 <= article_id < len(articles): fail('Stage2 article_id out of bounds')
        a = articles[article_id]
        if int(row['doc_id']) != a['doc_id']: fail('Stage2 doc ID mismatch')
        seen_articles.add(article_id)
        ref_ids = [int(x) for x in row['reference_sentence_ids']]
        hyp_ids = [int(x) for x in row['hypothesis_sentence_ids']]
        if not ref_ids or not hyp_ids: fail('Unexpected empty Stage2 aligned group')
        if any(x < 0 or x >= len(a['sentences']) for x in ref_ids): fail('Stage2 ref ID out of bounds')
        if any(x < 0 or x >= len(a['hypothesis_sentences']) for x in hyp_ids): fail('Stage2 hyp ID out of bounds')
        expected_ref = ' '.join(a['sentences'][x] for x in ref_ids)
        expected_hyp = ' '.join(a['hypothesis_sentences'][x] for x in hyp_ids)
        if expected_ref != row['reference'] or expected_hyp != row['hypothesis']:
            fail(f'Stage2 text mismatch in article {article_id}')
        src_ids_by_segment = collections.defaultdict(set)
        ref_keys = []
        for rid in ref_ids:
            loc = a['reference_local_keys'][rid]
            seg, sid = loc['segment_id'], loc['sentence_id']
            ref_keys.append({'segment_id': seg, 'sentence_id': sid})
            src_ids_by_segment[seg].update(source_lookup.get((a['doc_id'], seg, sid), set()))
        source_keys = [{'segment_id': seg, 'sentence_id': sid}
                       for seg in sorted(src_ids_by_segment)
                       for sid in sorted(src_ids_by_segment[seg])]
        if not source_keys:
            stats['stage2_groups_without_stage1_source'] += 1
        source_text = ' '.join(src_by_key[(a['doc_id'], x['segment_id'])][x['sentence_id']] for x in source_keys)
        result.append({'article_id': article_id, 'doc_id': a['doc_id'], 'segment_id': a['segment_id'],
                       'source_sentence_keys': source_keys, 'reference_sentence_keys': ref_keys,
                       'reference_sentence_ids': ref_ids, 'hypothesis_sentence_ids': hyp_ids,
                       'source': source_text, 'reference': expected_ref, 'hypothesis': expected_hyp,
                       'alignment_method': row['alignment_method']})
        stats['aligned_groups'] += 1
        stats['source_sentences_in_groups'] += len(source_keys)
        stats['hypothesis_sentences_in_groups'] += len(hyp_ids)
    write_jsonl(output / 'source_hypothesis_alignment_all.jsonl', result)
    usable = [r for r in result if r['source_sentence_keys'] and r['hypothesis'].strip()]
    write_jsonl(output / 'source_hypothesis_alignment.jsonl', usable)
    ref_covered = collections.defaultdict(set)
    hyp_covered = collections.defaultdict(set)
    for r in result:
        ref_covered[r['article_id']].update(r['reference_sentence_ids'])
        hyp_covered[r['article_id']].update(r['hypothesis_sentence_ids'])
    total_ref = sum(len(a['sentences']) for a in articles)
    total_hyp = sum(len(a['hypothesis_sentences']) for a in articles)
    report = {
        'setting': setting, 'documents': 171, 'articles': len(articles),
        'stage1_alignment_groups': len(stage1), 'stage2_alignment_groups': len(stage2),
        'articles_with_stage2_alignment': len(seen_articles),
        'source_hypothesis_usable_groups': len(usable),
        'stage2_groups_without_stage1_source': stats['stage2_groups_without_stage1_source'],
        'reference_sentences_total': total_ref,
        'reference_sentences_covered': sum(map(len, ref_covered.values())),
        'hypothesis_sentences_total': total_hyp,
        'hypothesis_sentences_covered': sum(map(len, hyp_covered.values())),
        'unaligned_reference_sentences': total_ref - sum(map(len, ref_covered.values())),
        'unaligned_hypothesis_sentences': total_hyp - sum(map(len, hyp_covered.values())),
    }
    (output / 'alignment_stats.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(
        description="WMT24 en-zh reference-hypothesis sentence alignment."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="MODE=JSONL",
        help="Repeat for single, segment, mturn, or sourceprime.",
    )
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--bleualign", type=pathlib.Path, required=True)
    parser.add_argument(
        "--stage1-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1]
        / "stage1_alignment",
    )
    parser.add_argument("--processes", type=int, default=1)
    args = parser.parse_args()

    modes = {
        "single": "single_turn",
        "segment": "segment",
        "mturn": "multi_turn",
        "sourceprime": "source_primed",
    }

    global INPUTS, HIST, PYTHON, BLEUALIGN
    INPUTS = {}

    for item in args.input:
        mode, separator, filename = item.partition("=")
        if not separator or mode not in modes or not filename:
            parser.error(f"Invalid input specification: {item}")

        setting = modes[mode]
        if setting in INPUTS:
            parser.error(f"Duplicate mode: {mode}")

        input_path = pathlib.Path(filename).expanduser().resolve()
        if not input_path.is_file():
            parser.error(f"Input file not found: {input_path}")
        INPUTS[setting] = str(input_path)

    if args.processes < 1:
        parser.error("--processes must be positive.")

    HIST = args.stage1_dir.expanduser().resolve()
    PYTHON = sys.executable
    bleualign = args.bleualign.expanduser().resolve()
    if not bleualign.is_file():
        parser.error(f"Bleualign entry point not found: {bleualign}")
    BLEUALIGN = str(bleualign)

    for filename in (
        "en-zh_source_sentences.jsonl",
        "en-zh_reference_sentences.jsonl",
        "stage1_alignment_structured.jsonl",
    ):
        if not (HIST / filename).is_file():
            parser.error(f"Missing Stage 1 file: {HIST / filename}")

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        parser.error("Output directory already exists; choose a new directory.")
    output_root.mkdir(parents=True, exist_ok=False)

    print(f"OUTPUT_ROOT={output_root}", flush=True)
    for setting in INPUTS:
        work = output_root / setting
        articles, stage1, source_lookup = prepare(setting, work)
        stage2 = run_bleualign(work, args.processes)
        report = compose(
            setting, work, articles, stage1, source_lookup, stage2
        )
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

    print(f"DONE: {output_root}", flush=True)


if __name__ == '__main__':
    main()
