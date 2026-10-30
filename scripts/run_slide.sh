#!/usr/bin/env bash
# Evaluate one translation file using the active SLIDE environment.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage:
  bash scripts/run_slide.sh \
    GPU_ID MODE INPUT_JSONL OUTPUT_DIR [segment|character|both]

Modes: single, segment, mturn, sourceprime

Default metric:
  single: character
  other modes: both

Single translation does not run segment-level SLIDE.
Existing output files are never overwritten.
HELP
}

if [[ "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if (( $# < 4 || $# > 5 )); then
    usage >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU=$1
MODE=$2
INPUT=$3
OUTDIR=$4
METRIC="${5:-}"

case "$MODE" in
    single) METRIC="${METRIC:-character}" ;;
    segment|mturn|sourceprime) METRIC="${METRIC:-both}" ;;
    *) echo "ERROR: Unknown mode: $MODE" >&2; exit 2 ;;
esac

case "$METRIC" in
    segment|character|both) ;;
    *) echo "ERROR: Unknown metric: $METRIC" >&2; exit 2 ;;
esac

if [[ "$MODE" == single && "$METRIC" != character ]]; then
    echo "ERROR: Single translation only supports character-level SLIDE." >&2
    exit 2
fi
[[ -f "$INPUT" ]] || {
    echo "ERROR: Input file not found: $INPUT" >&2
    exit 2
}

PYTHON="${PYTHON:-python}"
SCORER="${SLIDE_MODEL:-Unbabel/wmt22-cometkiwi-da}"
NAME="$(basename "$INPUT")"
NAME="${NAME%.jsonl}"

SEG_LOG="$OUTDIR/${NAME}_slide_segment.log"
CHAR_LOG="$OUTDIR/${NAME}_slide_character.log"
CHAR_JSON="$OUTDIR/${NAME}_slide_character.json"

# Check all selected outputs before starting either evaluation.
if [[ "$METRIC" == segment || "$METRIC" == both ]]; then
    [[ ! -e "$SEG_LOG" ]] || {
        echo "ERROR: Existing output: $SEG_LOG" >&2
        exit 2
    }
fi
if [[ "$METRIC" == character || "$METRIC" == both ]]; then
    [[ ! -e "$CHAR_LOG" && ! -e "$CHAR_JSON" ]] || {
        echo "ERROR: Character-level outputs already exist." >&2
        exit 2
    }
fi

mkdir -p "$OUTDIR"

if [[ "$METRIC" == segment || "$METRIC" == both ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u eval/eval_slide_small.py \
        "$INPUT" \
        --model "$SCORER" \
        --window-size 6 \
        --stride 6 \
        --batch-size 1 \
        --gpus 1 \
        2>&1 | tee "$SEG_LOG"
fi

if [[ "$METRIC" == character || "$METRIC" == both ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u \
        eval/eval_slide_small_character.py \
        "$INPUT" \
        --model "$SCORER" \
        --window-size 200 \
        --stride 100 \
        --batch-size 1 \
        --gpus 1 \
        --output-json "$CHAR_JSON" \
        2>&1 | tee "$CHAR_LOG"
fi

echo "Completed: $INPUT"
