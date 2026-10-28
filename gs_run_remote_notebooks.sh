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

TRAINING_JOBS=(splits small_cnn convnext vit swin efficientnet)
ALL_JOBS=(splits small_cnn convnext vit swin efficientnet ensemble)
CHANNEL_SETUPS=(RGB_only RGB_slope RGB_elevation RGB_slope_elevation)

resolve_channel_setups() {
    if [[ "${BEACH_CHANNEL_SETUP+x}" == x ]]; then
        case "${BEACH_CHANNEL_SETUP}" in
            RGB_only|RGB_slope|RGB_elevation|RGB_slope_elevation)
                CHANNEL_SETUPS=("${BEACH_CHANNEL_SETUP}")
                ;;
            *)
                echo "Invalid BEACH_CHANNEL_SETUP: '${BEACH_CHANNEL_SETUP}'." >&2
                echo "Choose RGB_only, RGB_slope, RGB_elevation, or RGB_slope_elevation." >&2
                exit 2
                ;;
        esac
    fi
}

timestamp() {
    date '+%Y-%m-%dT%H:%M:%S%z'
}

notebook_for_job() {
    case "$1" in
        splits) echo "gs_create_dataset_splits.ipynb" ;;
        small_cnn) echo "gs_train_small_cnn_classifier.ipynb" ;;
        convnext) echo "gs_train_convnext_classifier.ipynb" ;;
        vit) echo "gs_train_vit_tiny_classifier.ipynb" ;;
        swin) echo "gs_train_swin_tiny_classifier.ipynb" ;;
        efficientnet) echo "gs_train_efficientnetv2_classifier.ipynb" ;;
        ensemble) echo "gs_evaluate_ensemble.ipynb" ;;
        *) return 1 ;;
    esac
}

usage() {
    cat <<'EOF'
Usage:
  ./gs_run_remote_notebooks.sh all
  ./gs_run_remote_notebooks.sh training
  ./gs_run_remote_notebooks.sh convnext vit swin efficientnet
  ./gs_run_remote_notebooks.sh ensemble

Names: splits, small_cnn, convnext, vit, swin, efficientnet, ensemble

Training and ensemble jobs run for RGB_only, RGB_slope, RGB_elevation, and
RGB_slope_elevation by default. Set BEACH_CHANNEL_SETUP to run just one setup.
Splits run once, before the selected channel-dependent jobs.

The command starts a detached sequential grain-size pipeline. Each notebook
receives its own debug log and executed notebook. Use BEACH_PYTHON to select
another interpreter.
The all/training modes first generate the shared nested cross-validation plan.
Every model then evaluates five outer folds and fits a separate deployment model.
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

run_notebook() (
    local job="$1"
    local setup="${2:-}"
    local notebook
    notebook="$(notebook_for_job "${job}")"
    local stem="${notebook%.ipynb}"
    local log_path="${PROJECT_DIR}/${stem}_debug.log"
    local output_name="executed_notebooks/${stem}_executed.ipynb"
    if [[ "${job}" == splits ]]; then
        unset BEACH_CHANNEL_SETUP
    else
        export BEACH_CHANNEL_SETUP="${setup}"
        log_path="${PROJECT_DIR}/${stem}_${setup}_debug.log"
        output_name="executed_notebooks/${setup}/${stem}_executed.ipynb"
        mkdir -p "${EXECUTED_DIR}/${setup}"
    fi

    echo "[$(timestamp)] Starting ${job} (${setup:-channel-independent}): ${notebook}"
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
)

if [[ "${1:-}" != "--worker" ]]; then
    if [[ $# -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
        usage
        exit 0
    fi
    resolve_jobs "$@"
    resolve_channel_setups
    mkdir -p "${EXECUTED_DIR}"
    pipeline_log="${PROJECT_DIR}/gs_notebook_pipeline_$(date +%Y%m%d_%H%M%S).log"
    nohup "${BASH}" "${SCRIPT_PATH}" --worker "${RESOLVED_JOBS[@]}" \
        > "${pipeline_log}" 2>&1 < /dev/null &
    pipeline_pid=$!
    echo "Started grain-size notebook pipeline with PID ${pipeline_pid}."
    echo "Pipeline log: ${pipeline_log}"
    echo "Jobs: ${RESOLVED_JOBS[*]}"
    echo "Channel setups: ${CHANNEL_SETUPS[*]}"
    exit 0
fi

shift
resolve_jobs "$@"
resolve_channel_setups
mkdir -p "${EXECUTED_DIR}"
cd "${PROJECT_DIR}"

echo "[$(timestamp)] Grain-size pipeline started: ${RESOLVED_JOBS[*]}"
for job in "${RESOLVED_JOBS[@]}"; do
    if [[ "${job}" == splits ]]; then
        run_notebook splits
        break
    fi
done
for setup in "${CHANNEL_SETUPS[@]}"; do
    for job in "${RESOLVED_JOBS[@]}"; do
        if [[ "${job}" != splits ]]; then
            run_notebook "${job}" "${setup}"
        fi
    done
done
echo "[$(timestamp)] Grain-size pipeline completed successfully."
