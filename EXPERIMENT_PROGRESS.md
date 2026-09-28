# Experiment Progress — en-es / en-ca

Updated: 2026-09-24 Europe/Zurich

## Audit

- Repository: /local/scratch/yicsun/multiturn-llm-docmt_tine
- en-es: 171 documents, 998 source segments, 998 reference segments, 0 mismatched documents
- en-ca: 171 documents, 998 source segments, 998 reference segments, 0 mismatched documents
- Prompt directions verified: English→Spanish and English→Catalan
- translate.py syntax: PASS
- GPUs at audit: 0–7 free

## Smoke tests

| Condition | Status | GPUs | PID | Log | Output |
|---|---|---|---:|---|---|
| Llama en-es Segment | RUNNING | 0,1,2,3 | 1559229 | logs/codex_runs/llama_en-es_segment_smoke2.log | outputs/controlled_en_es_en_ca/llama_en-es_segment_smoke2_gen.jsonl |
| Qwen en-ca Multi-turn | RUNNING | 4,5,6,7 | 1559230 | logs/codex_runs/qwen_en-ca_mturn_smoke2.log | outputs/controlled_en_es_en_ca/qwen_en-ca_mturn_smoke2_gen.jsonl |
| Gemma Source V2 compatibility | PENDING | — | — | — | — |

## Full generation matrix

| Language | Model | Setting | Status | GPUs | PID | Log | Output |
|---|---|---|---|---|---:|---|---|
| en-es | Llama-3.1-8B-Instruct | Segment | PENDING | — | — | — | — |
| en-es | Llama-3.1-8B-Instruct | Multi-turn | PENDING | — | — | — | — |
| en-es | Llama-3.1-8B-Instruct | Source V2 | PENDING | — | — | — | — |
| en-es | Qwen2.5-7B-Instruct | Segment | PENDING | — | — | — | — |
| en-es | Qwen2.5-7B-Instruct | Multi-turn | PENDING | — | — | — | — |
| en-es | Qwen2.5-7B-Instruct | Source V2 | PENDING | — | — | — | — |
| en-es | Gemma-7B-it | Segment | PENDING | — | — | — | — |
| en-es | Gemma-7B-it | Multi-turn | PENDING | — | — | — | — |
| en-es | Gemma-7B-it | Source V2 | PENDING | — | — | — | — |
| en-ca | Llama-3.1-8B-Instruct | Segment | PENDING | — | — | — | — |
| en-ca | Llama-3.1-8B-Instruct | Multi-turn | PENDING | — | — | — | — |
| en-ca | Llama-3.1-8B-Instruct | Source V2 | PENDING | — | — | — | — |
| en-ca | Qwen2.5-7B-Instruct | Segment | PENDING | — | — | — | — |
| en-ca | Qwen2.5-7B-Instruct | Multi-turn | PENDING | — | — | — | — |
| en-ca | Qwen2.5-7B-Instruct | Source V2 | PENDING | — | — | — | — |
| en-ca | Gemma-7B-it | Segment | PENDING | — | — | — | — |
| en-ca | Gemma-7B-it | Multi-turn | PENDING | — | — | — | — |
| en-ca | Gemma-7B-it | Source V2 | PENDING | — | — | — | — |

## Evaluation

All 54 metric evaluations are PENDING until generation and structural validation complete.

## Live detached run (2026-09-24)

Smoke checks passed for Llama En�Es Segment, Qwen En�Ca Multi-turn, and Gemma En�Es Source-primed V2. Master PID 1584185 is running two server-resident queues. See controlled_en_es_en_ca_status.tsv and logs/codex_runs/controlled_master.log.
