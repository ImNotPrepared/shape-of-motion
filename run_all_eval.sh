#!/bin/bash

# Directory containing the folders
dir="/data3/zihanwa3/Capstone-DSR/shape-of-motion/results_dance"

# Loop over each folder that starts with "1_"
for folder in "$dir"/*; do
    # Get the base name of the folder
    exp=$(basename "$folder")
    # Run the python command with the folder name as the --exp argument
    python eval_final.py --seq_name 'dance' --exp "$exp"
done
