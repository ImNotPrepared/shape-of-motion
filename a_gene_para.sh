#! /usr/bin/env python
#rm -rf /data3/zihanwa3/Capstone-DSR/shape-of-motion/output_duster_feature_rendering
EXP=$1
sequences=(
    "_iiith_cooking_123_2"
    "_indiana_music_11_2"
    "_nus_cpr_08_1"
    "_unc_basketball_03-16-23_01_18"
)

# Explicit GPU ID list
gpu_ids=(2 3 6 7)

# Number of available GPUs
num_gpus=${#gpu_ids[@]}

# Loop through the sequences and run in parallel on different GPUs
for ((idx=0; idx<${#sequences[@]}; idx++))
do
    gpu_id=${gpu_ids[$(( idx % num_gpus ))]}  # Assign GPU from the list in a round-robin fashion
    seq="${sequences[idx]}"
    CUDA_VISIBLE_DEVICES=$gpu_id python dance_glb.py --seq_name "$seq" --exp "$EXP" &
done

# Wait for all background jobs to complete
wait

echo "All sequences processed."
