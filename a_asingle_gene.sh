#! /usr/bin/env python
#rm -rf /data3/zihanwa3/Capstone-DSR/shape-of-motion/output_duster_feature_rendering
EXP=$1
# _unc_basketball_03-16-23_01_18
# _indiana_music_11_2
# _iiith_cooking_123_2

# _nus_cpr_08_1

###  modest 
python dance_glb.py --seq_name '_panoptic_softball' --depth_type 'panoptic_gt' --exp "$EXP"