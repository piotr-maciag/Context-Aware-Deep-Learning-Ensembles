#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
PROJECT_DIR="${SCRIPT_DIR}"
if [[ ! -d "${PROJECT_DIR}/Notebooks" || ! -d "${PROJECT_DIR}/Source" ]]; then
    echo "Run this launcher from the remote project containing Notebooks and Source." >&2
    exit 2
fi

NOTEBOOK_DIR="${PROJECT_DIR}/Notebooks"
EXECUTED_DIR="${NOTEBOOK_DIR}/executed_notebooks"
BEACH_PYTHON="${BEACH_PYTHON:-python}"

TRAINING_JOBS=(small_cnn convnext vit swin efficientnet)
ALL_JOBS=(small_cnn convnext vit swin efficientnet ensemble)

timestamp() {
    date '+%Y-%m-%dT%H:%M:%S%z'
}

notebook_for_job() {
    case "$1" in
        small_cnn) echo "train_small_cnn_classifier.ipynb" ;;
        convnext) echo "train_convnext_classifier.ipynb" ;;
        vit) echo "train_vit_tiny_classifier.ipynb" ;;
        swin) echo "train_swin_tiny_classifier.ipynb" ;;
        efficientnet) echo "train_efficientnetv2_classifier.ipynb" ;;
        ensemble) echo "evaluate_weighted_ensemble.ipynb" ;;
        *) return 1 ;;
    esac
}

usage() {
    cat <<'EOF'
Usage:
  ./run_remote_notebooks.sh all
  ./run_remote_notebooks.sh training
  ./run_remote_notebooks.sh convnext vit swin efficientnet
  ./run_remote_notebooks.sh ensemble

Names: small_cnn, convnext, vit, swin, efficientnet, ensemble

The command starts a detached sequential pipeline. Each notebook receives its own
debug log and executed notebook. Use BEACH_PYTHON to select another interpreter.
EOF
}

resolve_jobs() {
    local requested=("$@")
    if [[ ${#requested[@]} -eq 1 && "${requested[0]}" == "all" ]]; then
        RESOLVED_JOBS=("${ALL_JOBS[@]}")
        return
    fi
    if [[ ${#requested[@]} -eq 1 && "${requested[0]}" == "training" ]]; then
        RESOLVED_JOBS=("${TRAINING_JOBS[@]}")
        return
    fi

    RESOLVED_JOBS=()
    local job
    for job in "${requested[@]}"; do
        if ! notebook_for_job "${job}" > /dev/null; then
            echo "Unknown job: ${job}" >&2
            usage >&2
            exit 2
        fi
        RESOLVED_JOBS+=("${job}")
    done
}

run_notebook() {
    local job="$1"
    local notebook
    notebook="$(notebook_for_job "${job}")"
    local stem="${notebook%.ipynb}"
    local log_path="${PROJECT_DIR}/${stem}_debug.log"
    local output_name="executed_notebooks/${stem}_executed.ipynb"

    echo "[$(timestamp)] Starting ${job}: ${notebook}"
    if "${BEACH_PYTHON}" -u -m jupyter nbconvert \
        --debug \
        --to notebook \
        --execute "${NOTEBOOK_DIR}/${notebook}" \
        --output "${output_name}" \
        --output-dir "${NOTEBOOK_DIR}" \
        --ExecutePreprocessor.timeout=-1 \
        > "${log_path}" 2>&1; then
        echo "[$(timestamp)] Finished ${job}; log: ${log_path}"
    else
        local exit_code=$?
        echo "[$(timestamp)] Failed ${job}; inspect: ${log_path}" >&2
        return "${exit_code}"
    fi
}

if [[ "${1:-}" != "--worker" ]]; then
    if [[ $# -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
        usage
        exit 0
    fi
    resolve_jobs "$@"
    mkdir -p "${EXECUTED_DIR}"
    pipeline_log="${PROJECT_DIR}/notebook_pipeline_$(date +%Y%m%d_%H%M%S).log"
    nohup "${BASH}" "${SCRIPT_PATH}" --worker "${RESOLVED_JOBS[@]}" \
        > "${pipeline_log}" 2>&1 < /dev/null &
    pipeline_pid=$!
    echo "Started notebook pipeline with PID ${pipeline_pid}."
    echo "Pipeline log: ${pipeline_log}"
    echo "Jobs: ${RESOLVED_JOBS[*]}"
    exit 0
fi

shift
resolve_jobs "$@"
mkdir -p "${EXECUTED_DIR}"
cd "${PROJECT_DIR}"

echo "[$(timestamp)] Pipeline started: ${RESOLVED_JOBS[*]}"
for job in "${RESOLVED_JOBS[@]}"; do
    run_notebook "${job}"
done
echo "[$(timestamp)] Pipeline completed successfully."
