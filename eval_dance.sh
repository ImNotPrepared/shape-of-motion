#! /usr/bin/env python
#rm -rf /data3/zihanwa3/Capstone-DSR/shape-of-motion/output_duster_feature_rendering
EXP=$1
combinations=(
  "2 3 0"
)
python eval_final.py --seq_name 'dance' --exp "$EXP" --train_indices ${combinations[i]}