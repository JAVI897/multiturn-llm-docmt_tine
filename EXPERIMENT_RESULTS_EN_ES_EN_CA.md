# Experiment Results — en-es / en-ca

## Experiment Overview

Controlled reproduction across Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, and Gemma-7B-it; settings Segment, Multi-turn, Source V2; metrics d-BLEU, Doc-COMET, and SLIDE-small.

## Dataset Statistics

| Direction | Source | Reference | Documents | Source segments | Reference segments | Mismatched documents |
|---|---|---|---:|---:|---:|---:|
| en-es | wmt24_processed/wmt24_en-es.en.txt | wmt24_processed/wmt24_en-es.es.txt | 171 | 998 | 998 | 0 |
| en-ca | wmt24pp_processed/wmt24pp_en-ca.en.txt | wmt24pp_processed/wmt24pp_en-ca.ca.txt | 171 | 998 | 998 | 0 |

## English → Spanish Results

| Model | Setting | d-BLEU | Doc-COMET | SLIDE-small |
|---|---|---:|---:|---:|

## English → Catalan Results

| Model | Setting | d-BLEU | Doc-COMET | SLIDE-small |
|---|---|---:|---:|---:|

## d-BLEU Diagnostics

Pending.

## SLIDE-small Diagnostics

Pending.

## Generation Validation and Qualitative Observations

Pending full generation.

## Limitations

- SLIDE-small uses Unbabel/wmt22-cometkiwi-da, not COMETKiwi-XXL.
- Doc-COMET uses the existing wmt21-comet-mqm checkpoint and repository protocol.
- No statistical significance claims are made.
- Model-specific instruction-following behavior may influence automatic metrics.
