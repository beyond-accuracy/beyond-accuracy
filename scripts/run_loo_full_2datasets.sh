#!/usr/bin/env bash
# LOO audit of the five models on the AV2 and Waymo validation sets (GPU 0 and 1 in parallel).
# Waymo uses the SDC as the focal agent (--sdc-only), as in the paper.
# Each output stores, per (scene, focal), the signed change of minADE6 / minFDE6 / brier-minFDE6
# when one surrounding agent is masked.
#
# Outputs: results/loo/{av2,waymo}/{DeMo,forecast,wayformer,autobot,MTR}.pkl
# Wall time on two RTX A6000: about 10 h (AV2) + 15 h (Waymo).
set -euo pipefail
cd "$(dirname "$0")/.."

LOG_BASE=logs/loo
mkdir -p "$LOG_BASE"

# ── AV2 checkpoints ────────────────────────────────────────────────
AV2_DEMO=ckpts/argoverse2/DeMo_av2.ckpt
AV2_FORECAST=ckpts/argoverse2/forecast_mae_av2.ckpt
AV2_WAYFORMER=ckpts/argoverse2/wayformer_av2.ckpt
AV2_AUTOBOT=ckpts/argoverse2/autobot_av2.ckpt
AV2_MTR=ckpts/argoverse2/MTR_av2.ckpt

# ── Waymo (WOMD) checkpoints ──────────────────────────────────────────────────
WMD_DEMO=ckpts/WOMD/DeMo_WOMD.ckpt
WMD_FORECAST=ckpts/WOMD/Forecast_WOMD.ckpt
WMD_WAYFORMER=ckpts/WOMD/Wayformer_WOMD.ckpt
WMD_AUTOBOT=ckpts/WOMD/Autobot_WOMD.ckpt
WMD_MTR=ckpts/WOMD/MTR_WOMD.ckpt

run_one() {
    local DATASET=$1 MODEL=$2 CKPT=$3 GPU=$4 OUT=$5
    local LOG="${LOG_BASE}/${DATASET}_${MODEL}_$(date +%Y%m%d-%H%M%S).log"
    echo "===== [$(date)] [GPU $GPU] $DATASET / $MODEL — $LOG ====="
    CUDA_VISIBLE_DEVICES=$GPU python tools/experiments/loo_analysis.py \
        --model "$MODEL" --dataset "$DATASET" --ckpt "$CKPT" \
        --output "$OUT" --seed 42 $([ "$DATASET" = waymo ] && echo --sdc-only) \
        2>&1 | tee "$LOG"
    echo "===== [$(date)] [GPU $GPU] $DATASET / $MODEL — done rc=${PIPESTATUS[0]} ====="
}

run_pair() {
    local TAG=$1 D1=$2 M1=$3 C1=$4 O1=$5 D2=$6 M2=$7 C2=$8 O2=$9
    echo ""
    echo "############ [$(date)] $TAG: $D1/$M1 (GPU0) || $D2/$M2 (GPU1) ############"
    run_one "$D1" "$M1" "$C1" 0 "$O1" &
    PID0=$!
    run_one "$D2" "$M2" "$C2" 1 "$O2" &
    PID1=$!
    wait $PID0; RC0=$?
    wait $PID1; RC1=$?
    echo "############ [$(date)] $TAG done: GPU0 rc=$RC0  GPU1 rc=$RC1 ############"
    if [[ $RC0 -ne 0 || $RC1 -ne 0 ]]; then
        echo "FATAL: pair $TAG failed; aborting chain"
        exit 1
    fi
}

OUT_AV2=results/loo/av2
OUT_WMD=results/loo/waymo
mkdir -p "$OUT_AV2" "$OUT_WMD"

# ── AV2 (3 rounds) ───────────────────────────────────────────────────────────
run_pair "AV2-R1" \
    av2 DeMo     "$AV2_DEMO"      "$OUT_AV2/DeMo.pkl"      \
    av2 autobot  "$AV2_AUTOBOT"   "$OUT_AV2/autobot.pkl"
run_pair "AV2-R2" \
    av2 forecast "$AV2_FORECAST"  "$OUT_AV2/forecast.pkl"  \
    av2 wayformer "$AV2_WAYFORMER" "$OUT_AV2/wayformer.pkl"
echo "############ [$(date)] AV2-R3: MTR alone on GPU0 ############"
run_one av2 MTR "$AV2_MTR" 0 "$OUT_AV2/MTR.pkl"

# ── Waymo (3 rounds) ─────────────────────────────────────────────────────────
run_pair "WMD-R1" \
    waymo DeMo     "$WMD_DEMO"      "$OUT_WMD/DeMo.pkl"      \
    waymo autobot  "$WMD_AUTOBOT"   "$OUT_WMD/autobot.pkl"
run_pair "WMD-R2" \
    waymo forecast "$WMD_FORECAST"  "$OUT_WMD/forecast.pkl"  \
    waymo wayformer "$WMD_WAYFORMER" "$OUT_WMD/wayformer.pkl"
echo "############ [$(date)] WMD-R3: MTR alone on GPU0 ############"
run_one waymo MTR "$WMD_MTR" 0 "$OUT_WMD/MTR.pkl"

echo ""
echo "============================================================"
echo "[$(date)] ALL 10 LOO RUNS COMPLETED."
echo "  AV2  pkls: $OUT_AV2"
echo "  Waymo pkls: $OUT_WMD"
echo "============================================================"
