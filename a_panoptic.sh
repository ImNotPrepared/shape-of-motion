#! /usr/bin/env python
#rm -rf /data3/zihanwa3/Capstone-DSR/shape-of-motion/output_duster_feature_rendering
EXP=$1
# "_iiith_cooking_123_2"
# "_indiana_music_11_2"
# "_nus_cpr_08_1"
# "_unc_basketball_03-16-23_01_18"
# "_cmu_soccer_07_3"
# "_uniandes_ball_002_17"
# "_indiana_piano_14_4"

# _nus_cpr_08_1
# _cmu_bike_74_7
# _dance_mit_2_12

###  modest , 

### _panoptic_softball
### _uniandes_ball_002_17
### _indiana_piano_14_4
### _panoptic_softball
### _panoptic_tennis


python dance_glb.py --seq_name '_panoptic_softball' --depth_type 'panoptic_gt' --exp "$EXP"