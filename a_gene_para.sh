#! /usr/bin/env python
#rm -rf /data3/zihanwa3/Capstone-DSR/shape-of-motion/output_duster_feature_rendering
EXP=$1
sequences=(
  "_iiith_cooking_123_2"
  "_nus_cpr_08_1"
  "_cmu_soccer_07_3"
  # "_uniandes_ball_002_17"
  "_indiana_piano_14_4"
)

# Explicit GPU ID list
gpu_ids=(2 3 4 6)

# Number of available GPUs
num_gpus=${#gpu_ids[@]}

# Loop through the sequences and run in parallel on different GPUs
for ((idx=0; idx<${#sequences[@]}; idx++))
do
    gpu_id=${gpu_ids[$(( idx % num_gpus ))]}  # Assign GPU from the list in a round-robin fashion
    seq="${sequences[idx]}"
    CUDA_VISIBLE_DEVICES=$gpu_id python dance_glb.py --seq_name "$seq" --exp "$EXP" --depth_type 'moge' &
done

# Wait for all background jobs to complete
wait

echo "All sequences processed."
