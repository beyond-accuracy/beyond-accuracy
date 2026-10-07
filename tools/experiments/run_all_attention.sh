#!/usr/bin/env bash
# Attention Weight Extraction — 5 models × 2 benchmarks
#
#
# Usage:
#   bash tools/experiments/run_all_attention.sh [waymo|av2|both]
#   GPU_IDS="2,3" BATCH_SIZE=64 bash tools/experiments/run_all_attention.sh waymo
#
# Outputs:
#   results/attention/waymo/{DeMo,forecast,wayformer,autobot}.pkl   (MTR has no dense attention)
#   results/attention/av2/{DeMo,forecast,wayformer,autobot}.pkl

set -e

TARGET="${1:-both}"
SCRIPT="tools/experiments/extract_attention.py"
CKPT_DIR="${CKPT_DIR:-ckpts}"
LOG_DIR="logs/attention"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GPU_IDS="${GPU_IDS:-2,3}"
mkdir -p "$LOG_DIR"

# Parse GPU_IDS into array
IFS=',' read -ra GPUS <<< "$GPU_IDS"
N_GPUS=${#GPUS[@]}
echo "GPU_IDS=$GPU_IDS  ($N_GPUS GPUs)"
echo "BATCH_SIZE=$BATCH_SIZE  NUM_WORKERS=$NUM_WORKERS"

cd "$(dirname "$0")/../.."

# Run one model on a specific GPU in the background
run_single() {
    local model="$1"
    local dataset="$2"
    local ckpt="$3"
    local out="$4"
    local gpu="$5"
    local log="$6"
    local extra_args="$7"

    CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" \
        --model "$model" \
        --dataset "$dataset" \
        --ckpt "$ckpt" \
        --output "$out" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        $extra_args > "$log" 2>&1 &
}

# Run models in parallel (N_GPUS at a time), wait for each wave to finish
run_parallel() {
    local dataset="$1"
    shift
    local extra_args="$1"
    shift
    local -n models_ref=$1
    local -n ckpts_ref=$2

    local i=0
    local pids=()
    local labels=()

    for model in "${models_ref[@]}"; do
        ckpt="${ckpts_ref[$model]}"
        out="results/attention/${dataset}/${model}.pkl"
        log="$LOG_DIR/${dataset}_${model}.log"

        # Output dir uses 'av2', but unitraj config name is also 'av2'
        local ds_name="$dataset"

        if [ ! -f "$ckpt" ]; then
            echo "[SKIP] $model: checkpoint not found at $ckpt"
            continue
        fi

        mkdir -p "$(dirname "$out")"

        gpu="${GPUS[$((i % N_GPUS))]}"
        echo "[${dataset}/$model on GPU $gpu] → $out  (log: $log)"
        run_single "$model" "$ds_name" "$ckpt" "$out" "$gpu" "$log" "$extra_args"
        pids+=($!)
        labels+=("${dataset}/$model")
        i=$((i + 1))

        # When we've filled all GPUs, wait for this wave to finish
        if [ $((i % N_GPUS)) -eq 0 ]; then
            echo "  ... waiting for wave (${labels[*]})"
            for pid in "${pids[@]}"; do
                wait "$pid" || echo "  [WARN] pid $pid failed"
            done
            pids=()
            labels=()
        fi
    done

    # Wait for final wave
    if [ ${#pids[@]} -gt 0 ]; then
        echo "  ... waiting for final wave (${labels[*]})"
        for pid in "${pids[@]}"; do
            wait "$pid" || echo "  [WARN] pid $pid failed"
        done
    fi
}

run_waymo() {
    echo "=========================================="
    echo "  WAYMO Attention Extraction (SDC-only)"
    echo "=========================================="

    declare -A WAYMO_CKPTS=(
        ["DeMo"]="$CKPT_DIR/WOMD/DeMo_WOMD.ckpt"
        ["forecast"]="$CKPT_DIR/WOMD/Forecast_WOMD.ckpt"
        ["wayformer"]="$CKPT_DIR/WOMD/Wayformer_WOMD.ckpt"
        ["autobot"]="$CKPT_DIR/WOMD/Autobot_WOMD.ckpt"
    )
    # MTR excluded: uses local (neighborhood-restricted) attention,
    # dense agent-to-agent attention matrix not available.
    local WAYMO_MODELS=(DeMo forecast wayformer autobot)

    run_parallel waymo "--sdc-only" WAYMO_MODELS WAYMO_CKPTS
}

run_av2() {
    echo ""
    echo "=========================================="
    echo "  ARGOVERSE 2 Attention Extraction"
    echo "=========================================="

    declare -A AV2_CKPTS=(
        ["DeMo"]="$CKPT_DIR/argoverse2/DeMo_av2.ckpt"
        ["forecast"]="$CKPT_DIR/argoverse2/forecast_mae_av2.ckpt"
        ["wayformer"]="$CKPT_DIR/argoverse2/wayformer_av2"
        ["autobot"]="$CKPT_DIR/argoverse2/autobot_av2.ckpt"
    )
    # MTR excluded (local attention — no dense agent-to-agent matrix)
    local AV2_MODELS=(DeMo forecast wayformer autobot)

    run_parallel av2 "" AV2_MODELS AV2_CKPTS
}

case "$TARGET" in
    waymo)
        run_waymo
        ;;
    av2)
        run_av2
        ;;
    both|*)
        run_waymo
        run_av2
        ;;
esac

echo ""
echo "=========================================="
echo "  DONE. Results in results/attention/"
echo "=========================================="
for d in waymo av2; do
    if [ -d "results/attention/$d" ]; then
        echo "  $d/: $(ls results/attention/$d/ 2>/dev/null | wc -l) files"
    fi
done
