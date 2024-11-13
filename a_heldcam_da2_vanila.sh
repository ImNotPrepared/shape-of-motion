#!/bin/bash

# Array of individual combinations, represented as arrays of numbers
EXP=$1
combinations=(
  "2 3 0"
)

# Array of individual device IDs
id_devices=(
  0
)

# Run the training scripts in parallel with each combination
tasks=()
for i in {0..0}; do
  CUDA_VISIBLE_DEVICES=${id_devices[i]} python dance_glb.py --depth_type load_vanila_depth --seq_name 'dance' --train_indices ${combinations[i]} --exp "$EXP" &
  tasks+=("$!")
done


# Wait for all background processes to finish
for task in "${tasks[@]}"; do
  wait $task
done



