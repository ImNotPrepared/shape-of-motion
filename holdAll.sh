#!/bin/bash

# Array of individual combinations, represented as arrays of numbers
EXP=$1
combinations=(
  "1 2 3"
  "1 2 0"
  "1 3 0"
  "2 3 0"
)

# Array of individual device IDs
id_devices=(
  4
  5
  6
  7
)

# Run the training scripts in parallel with each combination
# --work-dir "$WORK_DIR" --port "$PORT" 
tasks=()
for i in {0..3}; do
  CUDA_VISIBLE_DEVICES=${id_devices[i]} sh custom.sh results_dance/YES_RGB_10x 2311 &
  tasks+=("$!")
done


# Wait for all background processes to finish
for task in "${tasks[@]}"; do
  wait $task
done



