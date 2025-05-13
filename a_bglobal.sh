#!/usr/bin/env bash
###############################################################################
# launch_dance_glb.sh  – parallel multi-GPU runner
# Usage:  ./launch_dance_glb.sh <EXP_PREFIX>
###############################################################################
set -e

EXP_PREFIX="$1"                          # first CLI arg
ts=$(date '+%b_%I%P' | sed 's/^0//')     # e.g. May_07pm    # e.g. May_07pm

# ---------- edit to taste ----------------------------------------------------
seqs=(
  "_indiana_piano_14_4"
  "_nus_cpr_08_1"
  "_cmu_bike_74_7"
  "_mit_dance_02_12"
  "_cmu_soccer_07_3"
  "_iiith_cooking_123_2"
)

gpus=(0 1 2 3 4 5)      # list *logical* GPU IDs you want to use
# ---------------------------------------------------------------------------

# optional safeguard: don’t launch more jobs than GPUs
if (( ${#seqs[@]} > ${#gpus[@]} )); then
  echo "Error: only ${#gpus[@]} GPUs for ${#seqs[@]} sequences" >&2
  exit 1
fi

for i in "${!seqs[@]}"; do
  seq="${seqs[$i]}"
  gpu="${gpus[$i]}"
  EXP="${EXP_PREFIX}_${ts}"      # keep runs distinct

  echo "→ GPU $gpu │ $seq │ $EXP"

  CUDA_VISIBLE_DEVICES="$gpu" \
    python dance_glb.py \
      --seq_name  "$seq" \
      --depth_type 'moge' \
      --exp        "$EXP" &             # ampersand → background job
done

wait  # blocks until all background jobs finish
echo "✅  All sequences completed."
