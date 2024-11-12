
from PIL import Image
import numpy as np
import os
import imageio
import torch
import cv2
from scipy.spatial.transform import Rotation as R
import pandas as pd

def get_preset_data(size):
      # Define the poses
      poses = [
          [0.374366, 1.441324, -0.057627, -0.19374, -0.754455, 0.594122, 0.200704],
          [-1.42594, 1.021887, -0.088919, 0.224576, -0.702463, 0.64663, -0.194888],
          [1.457692, -0.240018, -0.077916, -0.522571, -0.55499, 0.436684, 0.477716],
          [-1.547741, -1.348028, -0.099894, -0.7248, 0.265416, -0.236691, 0.590082]
      ]
      intrinsics = [
          [1764.094727, 1764.094727, 1919.5, 1079.5],
          [1774.26709, 1774.26709, 1919.5, 1079.5],
          [1764.426025, 1764.426025, 1919.5, 1079.5],
          [1783.065308, 1783.065308, 1919.5, 1079.5]
      ]
      #       3.         [1764.426025, 1764.426025, 1920.0, 1080.0],

      poses=poses[:]
      intrinsics= intrinsics[:]
      # depths = depths[:]
      # ims = ims[:]
      # Convert pose data (tx, ty, tz, qx, qy, qz, qw) to 4x4 transformation matrices
      def convert_to_matrix(pose):
          tx, ty, tz, qx, qy, qz, qw = pose
          rotation = R.from_quat([qx, qy, qz, qw])
          rotation_matrix = rotation.as_matrix()
          
          transformation_matrix = np.eye(4)
          transformation_matrix[:3, :3] = rotation_matrix
          transformation_matrix[:3, 3] = [tx, ty, tz]
          
          return transformation_matrix
      pose_matrices = [(convert_to_matrix(pose)) for pose in poses]

      # Convert the intrinsics to 3x3 matrix format
      def convert_intrinsics_to_matrix(intrinsics, size):
          fx, fy, cx, cy = intrinsics
          ratio = size/3840
          fx *= ratio
          fy *= ratio
          cx *= ratio
          cy *= ratio
          intrinsics_matrix = np.array([
              [fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]
          ])
          
          return intrinsics_matrix

      # Convert the list of intrinsics to 3x3 matrices
      intrinsics_matrices = [convert_intrinsics_to_matrix(intrinsics, size) for intrinsics in intrinsics]

      pose_matrices = np.array(pose_matrices)
      intrinsics_matrices = np.array(intrinsics_matrices)
      return pose_matrices, intrinsics_matrices

def get_preset_dance(size):
    seq='_dance'
    df = pd.read_csv(f'/data3/zihanwa3/Capstone-DSR/Processing{seq}/trajectory/gopro_calibs.csv')[:]
    # Define the poses
    poses = df[['tx_world_cam', 'ty_world_cam', 'tz_world_cam', 'qw_world_cam', 'qx_world_cam', 'qy_world_cam', 'qz_world_cam',  ]].values.tolist()
        # pose: [tx, ty, tz, qw, qx, qy, qz]
    # 3.  [1.457692, -0.240018, -0.077916, -0.522571, -0.55499, 0.436684, 0.477716],
    intrinsics = df[['image_width','image_height','intrinsics_0','intrinsics_1','intrinsics_2','intrinsics_3']].values.tolist()
    #       3.         [1764.426025, 1764.426025, 1920.0, 1080.0],

    poses=poses[:]
    intrinsics= intrinsics[:]
    # Convert pose data (tx, ty, tz, qx, qy, qz, qw) to 4x4 transformation matrices
    def convert_to_matrix(pose):
        tx, ty, tz, qx, qy, qz, qw = pose
        rotation = R.from_quat([qx, qy, qz, qw])
        rotation_matrix = rotation.as_matrix()
        
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = [tx, ty, tz]
        
        return transformation_matrix
    pose_matrices = [(convert_to_matrix(pose)) for pose in poses]


    # Convert the intrinsics to 3x3 matrix format
    def convert_intrinsics_to_matrix(intrinsics, size):
        _, _, fx, fy, cx, cy = intrinsics
        cx -= 0.5
        cy -= 0.5
        ratio = size/3840
        fx *= ratio
        fy *= ratio
        cx *= ratio
        cy *= ratio
        intrinsics_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        return intrinsics_matrix

    # Convert the list of intrinsics to 3x3 matrices
    intrinsics_matrices = [convert_intrinsics_to_matrix(intrinsics, size) for intrinsics in intrinsics]


    pose_matrices = np.array(pose_matrices)
    intrinsics_matrices = np.array(intrinsics_matrices)
    return pose_matrices, intrinsics_matrices