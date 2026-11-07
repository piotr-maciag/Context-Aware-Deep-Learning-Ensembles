#!/usr/bin/env bash
# Run: bash bs_town_neck_run_remote_notebooks.sh all
# Requires prepared Town Neck data/splits and the five saved BS RGB_only artifacts.
# Optional: PYTHON=/path/to/python TOWN_NECK_SOURCE_RESULTS=/path/to/Results
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
mkdir -p "$ROOT/Notebooks/executed_notebooks/bs_town_neck" "$ROOT/logs/bs_town_neck"
if [[ "${1:-all}" != "--worker" ]]; then
  pipeline_log="$ROOT/logs/bs_town_neck/launcher.log"
  nohup bash "$ROOT/bs_town_neck_run_remote_notebooks.sh" --worker "$@" > "$pipeline_log" 2>&1 < /dev/null &
  echo "Started batch PID $!; log: $pipeline_log"
  exit 0
fi
shift
# Bind nbconvert to this interpreter rather than a potentially different default kernel.
"$PYTHON" -m ipykernel install --user --name town-neck --display-name "Town Neck experiments"
if [[ $# -eq 0 || "${1:-}" == "all" ]]; then
  set -- small_cnn convnext vit swin efficientnet ensemble
fi
for task in "$@"; do
  case "$task" in
    small_cnn) notebook=bs_town_neck_train_small_cnn_classifier ;;
    convnext) notebook=bs_town_neck_train_convnext_classifier ;;
    vit) notebook=bs_town_neck_train_vit_tiny_classifier ;;
    swin) notebook=bs_town_neck_train_swin_tiny_classifier ;;
    efficientnet) notebook=bs_town_neck_train_efficientnetv2_classifier ;;
    ensemble) notebook=bs_town_neck_evaluate_ensemble ;;
    *) echo "Unknown task: $task" >&2; exit 2 ;;
  esac
  echo "$(date): starting $notebook"
  "$PYTHON" -u -m jupyter nbconvert --debug --to notebook --execute \
    "$ROOT/Notebooks/$notebook.ipynb" \
    --output "${notebook}_executed.ipynb" \
    --output-dir "$ROOT/Notebooks/executed_notebooks/bs_town_neck" \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=town-neck \
    > "$ROOT/logs/bs_town_neck/${notebook}.log" 2>&1
  echo "$(date): completed $notebook"
done
