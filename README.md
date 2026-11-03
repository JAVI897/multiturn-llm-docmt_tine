# Document Translation Experiments

This repository extends [Source-primed Multi-turn Conversation Helps Large Language Models Translate Documents](https://arxiv.org/abs/2503.10494) with open-weight model experiments, translation launchers, and document-level evaluation workflows.

Translation experiments cover English → Chinese, Spanish, and Catalan. The sentence-alignment and LTCR workflow currently supports the fixed WMT24 English–Chinese dataset only.

## Translation settings

| Setting | Description |
|---|---|
| `single` | Translate the complete document in one request. |
| `segment` | Translate each segment independently. |
| `mturn` | Translate segments sequentially, retaining previous source–translation turns. |
| `sourceprime` | Provide the complete source document as context before sequential multi-turn translation. |

Model-specific prompt handling and history management are implemented in `translate.py`. Generation limits, prompt flags, GPU selection, and model paths must be specified for each experiment.

## Repository structure

```text
agent.py                         vLLM loading, generation, and cleanup
translate.py                     Translation entry point
templates/                       Prompt examples
scripts/
  run_translation.sh             One translation configuration
  run_translation_batch.sh       Multiple translation modes
  run_slide.sh                   SLIDE-small evaluation
eval/
  eval.py                        BLEU and standard COMET
  eval_doc_comet.py               Doc-COMET
  eval_slide_small.py             Segment-window SLIDE-small
  eval_slide_small_character.py   Character-window SLIDE-small
  doccomet/                      Doc-COMET environment and revision records
  ltcr/                          Sentence alignment and LTCR
docs/
  environments/                  Environment snapshots
  experiments/                   Historical experiment records
patches/                         Bleualign patch and base revision
```

## Environments and hardware

Different tasks use separate Python environments. Dependency snapshots record installed packages; they are not a single installation specification for every task.

| Task | Environment record |
|---|---|
| Dedicated CUDA 12.8 generation environment | `docs/environments/requirements-qwen35-cu128-freeze.txt` |
| SLIDE-small | `docs/environments/requirements-slide-freeze.txt` |
| LTCR | `docs/environments/requirements-ltcr-freeze.txt` |
| Doc-COMET | `eval/doccomet/requirements-doccomet-freeze.txt` |

The dedicated generation environment was recorded with Python 3.11.2, PyTorch 2.11.0+cu128, Transformers 5.20.2, vLLM 0.26.0+cu128, and Triton 3.6.0. SLIDE and LTCR environments use Python 3.9.22.

Doc-COMET runs separately on glados with Python 3.9.22, PyTorch 1.13.1+cu117, Transformers 4.21.3, and PyTorch Lightning 1.6.4.

| Server | GPUs | CPU | RAM |
|---|---|---|---|
| habra | 8 × RTX 4090, approximately 24 GiB each | 2 × AMD EPYC 9654; 192 physical cores total | 755 GiB |
| glados | 8 × RTX 2080 Ti, 11 GiB each | 2 × Intel Xeon Gold 6230; 40 physical cores total | 376 GiB |

Both servers run Debian 12. Individual experiments use a subset of available GPUs.

Currently, generation and evaluations other than Doc-COMET run on habra. Early SLIDE experiments ran on glados.

## Data format

Translation inputs are UTF-8 text files:

- Blank lines separate documents.
- Newlines within a document separate segments.
- Source and reference files must have matching document and segment order.

Experiments use WMT24 processed data for English–Chinese and English–Spanish, and WMT24++ processed data for English–Catalan. Supply the actual source and reference paths explicitly.

Generated JSONL contains one record per document. Segment-based and multi-turn outputs also contain `translation_split` and `reference_split`. Whole-document `single` outputs do not provide these aligned segment lists.

## Translation

Activate the appropriate environment first. From the repository root:

```bash
bash scripts/run_translation.sh --help
bash scripts/run_translation_batch.sh --help
```

One configuration:

```bash
bash scripts/run_translation.sh \
  "$GPU_IDS" "$MODE" "$MODEL" "$SOURCE" "$REFERENCE" "$OUTPUT_JSON" \
  --lang_direction en-zh \
  --max_model_len "$CONTEXT_LENGTH" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
  --temperature 0
```

Set the variables to the intended experiment configuration. `OUTPUT_JSON` must end in `.json`; the generated result ends in `_gen.jsonl`. The launcher defaults to all documents. Use `--data_num` to override this.

Prompt flags such as `--is_og` and `--is_tower` must match the intended experiment. They are not automatically selected by the launcher.

The batch launcher runs selected modes with shared settings. Use it only when those settings are appropriate for every selected mode. Existing results and logs are not overwritten.

## BLEU and standard COMET

`eval/eval.py` contains the BLEU and standard COMET functions. Standard COMET uses `Unbabel/wmt22-comet-da`.

For the existing JSONL evaluation paths:

- BLEU skips the first JSONL record.
- COMET flattens segment lists and skips the first segment.

These operations assume the skipped entry is the WMT canary; the functions do not inspect its content. Do not use this behavior unchanged on inputs that have already removed the canary.

Whole-document `single` outputs cannot be passed directly to the segment-based COMET function.

## Doc-COMET

`eval/eval_doc_comet.py` uses the COMET implementation from:

https://github.com/amazon-science/doc-mt-metrics

The recorded revision is:

```text
5385cc28930aae9924edcb3201645dd3810b12c0
```

It loads the `wmt21-comet-mqm` checkpoint and requires aligned source, translation, and reference segment lists.

Context construction uses up to two preceding segments within the same document:

- Source context comes from preceding source segments.
- Reference context comes from preceding reference segments.
- Translation context comes from preceding reference segments, followed by the current hypothesis.

Configure `COMET_DIR` in the script for the external checkout. Supply the checkpoint explicitly:

```bash
python eval/eval_doc_comet.py \
  --input "$INPUT_JSONL" \
  --checkpoint "$CHECKPOINT" \
  --batch_size 8 \
  --gpus 1
```

## SLIDE-small

Both SLIDE variants use `Unbabel/wmt22-cometkiwi-da`. This smaller checkpoint was initially selected because of resource constraints on glados and retained for subsequent experiments on habra.

This differs from the original paper's evaluation configuration. Scores should not be treated as directly equivalent to the paper's reported results.

| Variant | Window | Stride | Translation modes |
|---|---:|---:|---|
| Segment-window | 6 segments | 6 segments | `segment`, `mturn`, `sourceprime` |
| Character-window | 200 source characters | 100 source characters | All four modes |

Segment-window evaluation retains complete windows only. Character-window evaluation maps source positions to hypothesis positions proportionally; this is approximate and does not guarantee semantic alignment.

```bash
bash scripts/run_slide.sh \
  "$GPU_ID" "$MODE" "$INPUT_JSONL" "$OUTPUT_DIR"
```

`single` runs character-window evaluation only. Other modes default to both variants. Across three language directions, the full matrix contains 9 segment-window and 12 character-window evaluations per model.

## Sentence alignment and LTCR

LTCR uses two alignment stages because the dataset is aligned at segment level, while each segment may contain multiple sentences.

### Stage 1: Source–reference alignment

English source sentences are translated with NLLB-200-3.3B to provide a Chinese bridge. Bleualign uses the bridge to establish source–reference sentence correspondences within each original segment.

This stage is shared across models and modes, provided the dataset, sentence splits, and identifiers remain unchanged.

Archived materials are stored in `eval/ltcr/stage1_alignment/`. The exact historical Stage 1 command has not been recovered; the current workflow reuses the archived structured alignment.

### Stage 2: Reference–hypothesis alignment

`eval/ltcr/sentence_alignment/run_sentence_alignment.py` aligns Chinese reference and hypothesis sentences:

- Within each document for `single`.
- Within each original segment for the other three modes.

It combines these correspondences with Stage 1 to obtain source–hypothesis groups, preserving document identifiers.

### Evaluation

The unified entry point performs Stage 2 alignment, composes the alignments, and runs document-bounded word alignment and LTCR scoring:

```bash
python eval/ltcr/run_ltcr_en_zh.py \
  --model "$MODEL_LABEL" \
  --input "$MODE=$INPUT_JSONL" \
  --bleualign "$BLEUALIGN_ENTRYPOINT" \
  --align-batch-size 8 \
  --processes 1 \
  --output "$RESULT_JSON"
```

Repeat `--input` for multiple modes. Model names are reporting labels, not a restricted list. Inputs must match the fixed WMT24 en-zh source data and document order.

The workflow requires patched Bleualign, awesome-align, its multilingual BERT checkpoint, and the tokenization dependencies. It sums exact matches, fuzzy matches, and eligible repeated-entity comparisons across documents to report LTCR-1 and LTCR-1f.

### Bleualign patch

Download a separate Bleualign checkout, select the recorded base revision, and apply the patch before first use:

```bash
TINE_ROOT="$PWD"
git clone https://github.com/rsennrich/Bleualign.git ../Bleualign
git -C ../Bleualign checkout \
  "$(cat "$TINE_ROOT/patches/bleualign-base-commit.txt")"
git -C ../Bleualign apply --check \
  "$TINE_ROOT/patches/bleualign-alignment-export.patch"
git -C ../Bleualign apply \
  "$TINE_ROOT/patches/bleualign-alignment-export.patch"
```

The patch adds structured alignment export, identifiers, checks, and diagnostic output. An existing patched checkout must not be patched again.

## Status and provenance

The reorganized launchers have passed syntax checks. The LTCR evaluator-loading adaptation has also been checked, but the reorganized pipeline has not yet been validated end to end. Some model evaluations remain pending.

Files in `docs/experiments/` describe historical runs and are not a current completion ledger.

LTCR-related code was adapted from:

https://github.com/YutongWang1216/DocMTAgent

The recorded upstream revision is `6dde70acc9e60fd2533c2ca467eb327884d20dab`. Upstream source, revision, and license records are stored in `eval/ltcr/upstream/`. Local adaptations include repository paths, unified model-independent launchers, and input handling.

## Citation

Please cite the original translation method when using it:

```bibtex
@misc{hu2025sourceprimedmultiturnconversationhelps,
  title={Source-primed Multi-turn Conversation Helps Large Language Models Translate Documents},
  author={Hanxu Hu and Jannis Vamvas and Rico Sennrich},
  year={2025},
  eprint={2503.10494},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2503.10494}
}
```

Also acknowledge the datasets, model checkpoints, and evaluation tools used in the experiment.