#!/usr/bin/env bash
# Run multiple translation modes with shared model and generation settings.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage:
  bash scripts/run_translation_batch.sh \
    GPU_IDS MODEL LANGUAGE SOURCE REFERENCE OUTPUT_DIR RUN_ID \
    [additional translate.py arguments]

Example LANGUAGE: en-zh
RUN_ID: a date or descriptive experiment identifier.

Environment variables:
  MODES   Space-separated modes; default:
          "single segment mturn sourceprime"
  PYTHON  Python executable; default: python

Use this launcher only when all selected modes share the supplied
prompt and generation settings. Run configurations separately otherwise.
HELP
}

if [[ "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if (( $# < 7 )); then
    usage >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU_IDS=$1
MODEL=$2
LANGUAGE=$3
SOURCE=$4
REFERENCE=$5
OUTDIR=$6
RUN_ID=$7
shift 7

case "$LANGUAGE" in
    en-zh|en-es|en-ca) ;;
    *) echo "ERROR: Unsupported language: $LANGUAGE" >&2; exit 2 ;;
esac

[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || {
    echo "ERROR: RUN_ID may contain letters, digits, underscores and hyphens." >&2
    exit 2
}
[[ -f "$SOURCE" && -f "$REFERENCE" ]] || {
    echo "ERROR: Source or reference file not found." >&2
    exit 2
}

# Keep language selection explicit and consistent across the batch.
for ARG in "$@"; do
    case "$ARG" in
        --lang_direction|--lang_direction=*)
            echo "ERROR: Specify the language using the LANGUAGE argument." >&2
            exit 2
            ;;
    esac
done

read -r -a SELECTED_MODES <<< "${MODES:-single segment mturn sourceprime}"
(( ${#SELECTED_MODES[@]} > 0 )) || {
    echo "ERROR: No modes selected." >&2
    exit 2
}

# Validate every output before starting the first configuration.
SEEN=" "
for MODE in "${SELECTED_MODES[@]}"; do
    case "$MODE" in
        single|segment|mturn|sourceprime) ;;
        *) echo "ERROR: Unknown mode: $MODE" >&2; exit 2 ;;
    esac
    if [[ "$SEEN" == *" $MODE "* ]]; then
        echo "ERROR: Duplicate mode: $MODE" >&2
        exit 2
    fi
    SEEN+="$MODE "

    BASE="$OUTDIR/${LANGUAGE}_${MODE}_${RUN_ID}"
    for FILE in "$BASE.json" "${BASE}_gen.jsonl" "$BASE.log"; do
        if [[ -e "$FILE" ]]; then
            echo "ERROR: Existing output: $FILE" >&2
            exit 2
        fi
    done
done

for MODE in "${SELECTED_MODES[@]}"; do
    OUTPUT="$OUTDIR/${LANGUAGE}_${MODE}_${RUN_ID}.json"
    bash "$ROOT/scripts/run_translation.sh" \
        "$GPU_IDS" "$MODE" "$MODEL" "$SOURCE" "$REFERENCE" "$OUTPUT" \
        --lang_direction "$LANGUAGE" \
        "$@"
done

echo "All selected translation configurations completed."
