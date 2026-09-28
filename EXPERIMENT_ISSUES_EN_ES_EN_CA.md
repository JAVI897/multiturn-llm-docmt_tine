# Experiment Issues — en-es / en-ca

## Issue 1 — en-ca data located outside wmt24_processed

Language: en-ca
Stage: Audit
Date: 2026-09-24

### Symptom
Expected wmt24_processed/wmt24_en-ca.* files were absent.

### Diagnosis
The validated en-ca corpus is WMT24++ and is stored as wmt24pp_processed/wmt24pp_en-ca.en.txt and wmt24pp_en-ca.ca.txt.

### Resolution
Use the actual WMT24++ paths. Counts were validated: 171 documents, 998 segments on both sides, 0 document mismatches.

### Methodological impact
Implementation/path correction only; no change to data or methodology.

### Outcome
Resolved.

### Invalid artifacts
None from this run phase.

## Issue 2 — FlashAttention 2 unsupported warning during Qwen smoke initialization

Language: en-ca
Model: Qwen2.5-7B-Instruct
Setting: Multi-turn
Stage: Smoke test
Date: 2026-09-24

### Symptom
vLLM logged that FA2 requires compute capability >= 8; RTX 2080 Ti is SM75.

### Diagnosis
Hardware/backend compatibility warning during initialization.

### Resolution
Pending final smoke outcome. No package or methodology change made.

### Methodological impact
None so far.

### Outcome
Under observation.

### Invalid artifacts
None unless the smoke run ultimately fails.

## Gemma native context limit

Gemma-7B-it rejected the repository default max_model_len=16384 because its checkpoint declares max_position_embeddings=8192. The safe compatibility setting --max_model_len 8192 is used for Gemma only; prompts, decoding settings, datasets, and method flags remain unchanged. The smoke test then completed successfully. Its shutdown emitted NCCL/shared-memory cleanup warnings after the output had been saved; these did not affect the generated JSONL.
