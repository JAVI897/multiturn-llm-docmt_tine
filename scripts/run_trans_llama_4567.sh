#!/usr/bin/env bash

# MODEL_NAME_PATH=Meta-Llama/Meta-Llama-3.1-8B-Instruct
# MODEL_NAME=Meta-Llama-3.1-8B-Instruct
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TORCH_USE_CUDA_DSA=1
SRC_LANG=$1
TGT_LANG=$2
SETTING=$3
DATA_PATH=$4
MODEL_NAME_PATH=$5
MODEL_NAME=$(basename $MODEL_NAME_PATH)
MAX_MODEL_LEN=${6:-16384}
mkdir -p outputs/${MODEL_NAME}
NUM=-1  # -1 means all data
DEVICES=4,5,6,7
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_LAUNCH_BLOCKING=1

if [ "$SETTING" == "mturn_icl" ]; then
  NCCL_DEBUG=INFOCUDA_VISIBLE_DEVICES="${DEVICES}" python3 translate.py \
    --source_file ${DATA_PATH}/wmt24_${SRC_LANG}-${TGT_LANG}.${SRC_LANG}.txt \
    --target_file ${DATA_PATH}/wmt24_${SRC_LANG}-${TGT_LANG}.${TGT_LANG}.txt \
    --model_name $MODEL_NAME_PATH \
    --output_file outputs/${MODEL_NAME}/wmt2024_${SRC_LANG}-${TGT_LANG}_mturn_icl.jsonl \
    --is_icl \
    --is_og \
    --is_tower \
    --is_conversation \
    --max_new_tokens 256 \
    --gpu_memory_utilization 0.9 \
    --lang_direction "${SRC_LANG}-${TGT_LANG}" \
    --data_num "${NUM}"
elif [ "$SETTING" == "seg_icl" ]; then
  NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES="${DEVICES}" python3 translate.py \
    --source_file ${DATA_PATH}/wmt24_${SRC_LANG}-${TGT_LANG}.${SRC_LANG}.txt \
    --target_file ${DATA_PATH}/wmt24_${SRC_LANG}-${TGT_LANG}.${TGT_LANG}.txt \
    --model_name $MODEL_NAME_PATH \
    --output_file outputs/${MODEL_NAME}/wmt2024_${SRC_LANG}-${TGT_LANG}_seg_icl.jsonl \
    --is_segment \
    --is_icl \
    --is_og \
    --is_tower \
    --max_new_tokens 256 \
    --gpu_memory_utilization 0.9 \
    --lang_direction "${SRC_LANG}-${TGT_LANG}" \
    --data_num "${NUM}"
elif [ "$SETTING" == "mturn" ]; then
  NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES="${DEVICES}" python3 translate.py \
    --source_file ${DATA_PATH}/wmt24_${SRC_LANG}-${TGT_LANG}.${SRC_LANG}.txt \
    --target_file ${DATA_PATH}/wmt24_${SRC_LANG}-${TGT_LANG}.${TGT_LANG}.txt \
    --model_name $MODEL_NAME_PATH \
    --output_file outputs/${MODEL_NAME}/wmt2024_${SRC_LANG}-${TGT_LANG}_mturn.jsonl \
    --is_conversation \
    --is_og \
    --is_tower \
    --max_new_tokens 256 \
    --gpu_memory_utilization 0.9 \
    --max_model_len "${MAX_MODEL_LEN}" \
    --lang_direction "${SRC_LANG}-${TGT_LANG}" \
    --data_num "${NUM}"
elif [ "$SETTING" == "seg" ]; then
  NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES="${DEVICES}" python3 translate.py \
    --source_file ${DATA_PATH}/wmt24_${SRC_LANG}-${TGT_LANG}.${SRC_LANG}.txt \
    --target_file ${DATA_PATH}/wmt24_${SRC_LANG}-${TGT_LANG}.${TGT_LANG}.txt \
    --model_name $MODEL_NAME_PATH \
    --output_file outputs/${MODEL_NAME}/wmt2024_${SRC_LANG}-${TGT_LANG}_seg.jsonl \
    --is_segment \
    --is_og \
    --is_tower \
    --max_new_tokens 256 \
    --gpu_memory_utilization 0.95 \
    --max_model_len "${MAX_MODEL_LEN}" \
    --lang_direction "${SRC_LANG}-${TGT_LANG}" \
    --data_num "${NUM}"
elif [ "$SETTING" == "single" ]; then
  NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES="${DEVICES}" python3 translate.py \
    --source_file ${DATA_PATH}/wmt24_${SRC_LANG}-${TGT_LANG}.${SRC_LANG}.txt \
    --target_file ${DATA_PATH}/wmt24_${SRC_LANG}-${TGT_LANG}.${TGT_LANG}.txt \
    --model_name $MODEL_NAME_PATH \
    --output_file outputs/${MODEL_NAME}/wmt2024_${SRC_LANG}-${TGT_LANG}_single.jsonl \
    --is_og \
    --is_tower \
    --max_new_tokens 2048 \
    --gpu_memory_utilization 0.95 \
    --lang_direction "${SRC_LANG}-${TGT_LANG}" \
    --data_num "${NUM}"
elif [ "$SETTING" == "mturn_context" ]; then
  NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES="${DEVICES}" python3 translate.py \
    --source_file ${DATA_PATH}/wmt24_${SRC_LANG}-${TGT_LANG}.${SRC_LANG}.txt \
    --target_file ${DATA_PATH}/wmt24_${SRC_LANG}-${TGT_LANG}.${TGT_LANG}.txt \
    --model_name $MODEL_NAME_PATH \
    --output_file outputs/${MODEL_NAME}/wmt2024_${SRC_LANG}-${TGT_LANG}_mturn_context.jsonl \
    --is_conversation \
    --is_og \
    --is_provide_all_first \
    --gpu_memory_utilization 0.95 \
    --max_model_len "${MAX_MODEL_LEN}" \
    --lang_direction "${SRC_LANG}-${TGT_LANG}" \
    --data_num "${NUM}"
elif [ "$SETTING" == "mturn_icl_context" ]; then
  NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES="${DEVICES}" python3 translate.py \
    --source_file ${DATA_PATH}/wmt24_${SRC_LANG}-${TGT_LANG}.${SRC_LANG}.txt \
    --target_file ${DATA_PATH}/wmt24_${SRC_LANG}-${TGT_LANG}.${TGT_LANG}.txt \
    --model_name $MODEL_NAME_PATH \
    --output_file outputs/${MODEL_NAME}/wmt2024_${SRC_LANG}-${TGT_LANG}_mturn_icl_context.jsonl \
    --is_conversation \
    --is_og \
    --is_provide_all_first \
    --is_icl \
    --gpu_memory_utilization 0.95 \
    --lang_direction "${SRC_LANG}-${TGT_LANG}" \
    --data_num "${NUM}"
elif [ "$SETTING" == "single_icl" ]; then
  NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES="${DEVICES}" python3 translate.py \
    --source_file ${DATA_PATH}/wmt24_${SRC_LANG}-${TGT_LANG}.${SRC_LANG}.txt \
    --target_file ${DATA_PATH}/wmt24_${SRC_LANG}-${TGT_LANG}.${TGT_LANG}.txt \
    --model_name $MODEL_NAME_PATH \
    --output_file outputs/${MODEL_NAME}/wmt2024_${SRC_LANG}-${TGT_LANG}_single_icl.jsonl \
    --is_icl \
    --is_og \
    --is_tower \
    --max_new_tokens 2048 \
    --gpu_memory_utilization 0.95 \
    --lang_direction "${SRC_LANG}-${TGT_LANG}" \
    --data_num "${NUM}"
fi