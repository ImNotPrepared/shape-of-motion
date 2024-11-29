#! /usr/bin/env python
#rm -rf /data3/zihanwa3/Capstone-DSR/shape-of-motion/output_duster_feature_rendering
EXP=$1
# results_nus_cpr_08_1
python eval_nnvs.py --seq_name '_nus_cpr_08_1' --exp "$EXP"