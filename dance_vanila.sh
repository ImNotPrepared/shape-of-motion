#! /usr/bin/env python
#rm -rf /data3/zihanwa3/Capstone-DSR/shape-of-motion/output_duster_feature_rendering
EXP=$1
python dance_glb.py --seq_name 'dance' --depth_type load_vanila_depth --exp "$EXP"
