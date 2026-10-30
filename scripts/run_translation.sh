#!/usr/bin/env bash
# Run one translation configuration using the active Python environment.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage:
  bash scripts/run_translation.sh \
    GPU_IDS MODE MODEL SOURCE REFERENCE OUTPUT_JSON \
    [additional translate.py arguments]

Modes: single, segment, mturn, sourceprime

OUTPUT_JSON must end in .json.
The generated file will end in _gen.jsonl.
Select the required Python environment before running.
Prompt flags and generation settings are passed explicitly.
HELP
}

if [[ "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if (( $# < 6 )); then
    usage >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU_IDS=$1
MODE=$2
MODEL=$3
SOURCE=$4
REFERENCE=$5
OUTPUT=$6
shift 6

FLAGS=()
case "$MODE" in
    single) ;;
    segment) FLAGS+=(--is_segment) ;;
    mturn) FLAGS+=(--is_conversation) ;;
    sourceprime) FLAGS+=(--is_conversation --is_provide_all_first) ;;
    *) echo "ERROR: Unknown mode: $MODE" >&2; exit 2 ;;
esac

[[ -f "$SOURCE" ]] || {
    echo "ERROR: Source file not found: $SOURCE" >&2
    exit 2
}
[[ -f "$REFERENCE" ]] || {
    echo "ERROR: Reference file not found: $REFERENCE" >&2
    exit 2
}
[[ "$OUTPUT" == *.json ]] || {
    echo "ERROR: Output must end in .json" >&2
    exit 2
}

# Prevent forwarded arguments from overriding wrapper-controlled settings.
for ARG in "$@"; do
    case "$ARG" in
        --model_name|--model_name=*|--source_file|--source_file=*|\
        --target_file|--target_file=*|--output_file|--output_file=*|\
        --is_segment|--is_segment=*|--is_conversation|--is_conversation=*|\
        --is_provide_all_first|--is_provide_all_first=*)
            echo "ERROR: Argument is controlled by this launcher: $ARG" >&2
            exit 2
            ;;
    esac
done

RESULT="${OUTPUT%.json}_gen.jsonl"
LOG="${OUTPUT%.json}.log"
if [[ -e "$OUTPUT" || -e "$RESULT" || -e "$LOG" ]]; then
    echo "ERROR: Output or log already exists; choose a new output name." >&2
    exit 2
fi
mkdir -p "$(dirname "$OUTPUT")"

PYTHON="${PYTHON:-python}"
echo "Model: $MODEL"
echo "Mode: $MODE"
echo "GPUs: $GPU_IDS"
echo "Result: $RESULT"

CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -u translate.py \
    --model_name "$MODEL" \
    --source_file "$SOURCE" \
    --target_file "$REFERENCE" \
    --output_file "$OUTPUT" \
    --data_num -1 \
    "${FLAGS[@]}" \
    "$@" \
    2>&1 | tee "$LOG"

[[ -s "$RESULT" ]] || {
    echo "ERROR: Expected nonempty result was not produced: $RESULT" >&2
    exit 1
}
echo "Completed: $RESULT"
