#!/bin/bash

# Directory containing the folders
dir="/data3/zihanwa3/Capstone-DSR/shape-of-motion/results_dance"
combinations=(
  "1 2 3"
  "1 2 0"
  "1 3 0"
  "2 3 0"
)

# Array of individual device IDs
id_devices=(
  2
  5
  6
  7
)

# Loop over each folder that starts with "1_"
tasks=()
for folder in "$dir"/1*; do
    # Get the base name of the folder
    exp=$(basename "$folder")
    for i in {0..3}; do
    # Run the python command with the folder name as the --exp argument
      CUDA_VISIBLE_DEVICES=${id_devices[i]} python eval_final.py --seq_name 'dance' --exp "$exp" --train_indices ${combinations[i]}
      tasks+=("$!")
    done
done

tasks=()
for folder in "$dir"/2*; do
    # Get the base name of the folder
    exp=$(basename "$folder")
    for i in {0..3}; do
    # Run the python command with the folder name as the --exp argument
      CUDA_VISIBLE_DEVICES=${id_devices[i]} python eval_final.py --seq_name 'dance' --exp "$exp" --train_indices ${combinations[i]}
      tasks+=("$!")
    done
done

tasks=()
for folder in "$dir"/3*; do
    # Get the base name of the folder
    exp=$(basename "$folder")
    for i in {0..3}; do
    # Run the python command with the folder name as the --exp argument
      CUDA_VISIBLE_DEVICES=${id_devices[i]} python eval_final.py --seq_name 'dance' --exp "$exp" --train_indices ${combinations[i]}
      tasks+=("$!")
    done
done


