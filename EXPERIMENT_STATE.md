# Experiment State â€” en-es / en-ca

Updated: 2026-09-24 Europe/Zurich

- Repository: /local/scratch/yicsun/multiturn-llm-docmt_tine
- Generation environment: /local/scratch/yicsun/multi_turn_vllm_0.10.2
- Doc-COMET environment: /local/scratch/yicsun/doc_comet_env
- SLIDE-small environment: /local/scratch/yicsun/slide_small_env
- Models:
  - /local/scratch/yicsun/hf-models/Llama-3.1-8B-Instruct
  - /local/scratch/yicsun/hf-models/Qwen2.5-7B-Instruct
  - /local/scratch/yicsun/hf-models/gemma-7b-it
- en-es data: wmt24_processed/wmt24_en-es.en.txt / wmt24_en-es.es.txt
- en-ca data: wmt24pp_processed/wmt24pp_en-ca.en.txt / wmt24pp_en-ca.ca.txt
- Frozen settings: Segment, Multi-turn, Source-primed V2
- Segment/Multi-turn use the validated --is_og --is_tower paths, max_new_tokens=256.
- Source V2 uses --is_conversation --is_og --is_provide_all_first and the validated context acknowledgement protocol.
- Gemma always requires VLLM_USE_V1=0 and VLLM_ATTENTION_BACKEND=XFORMERS.
- Qwen Source V2 uses gpu_memory_utilization=0.80.
- Logs: logs/codex_runs/
- Outputs: outputs/controlled_en_es_en_ca/
- Current jobs: smoke PIDs 1559229 and 1559230.

## 2026-09-24 controlled En’Es / En’Ca run

- Master PID: 1584185
- Detached launcher: run_controlled_en_es_en_ca.sh
- Status ledger: controlled_en_es_en_ca_status.tsv
- En’Es queue: GPUs 0,1,2,3
- En’Ca queue: GPUs 4,5,6,7
- Current state: Llama Segment generation running for both directions.
- The launcher covers all 18 generation conditions, then dBLEU, Doc-COMET, and SLIDE-small for every completed output.
