import os
import shutil
from PIL import Image
import cv2
import numpy as np

for index in range(1, 5):
    src_dir = f'/data3/zihanwa3/Capstone-DSR/shape-of-motion/data/masks/toy_{index}/'
    dest_dir = f'/data3/zihanwa3/Capstone-DSR/shape-of-motion/data/masks/toy_512_{index}/'

    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Define the range of image numbers to copy and resize
    start_index = 183
    end_index = 294

    # Loop through the range and copy each file, then resize
    for i in range(start_index, end_index + 1):
        src_file = os.path.join(src_dir, f'{i:05}.npz')
        dest_file = os.path.join(dest_dir, f'{i:05}.npz')
        
        if os.path.exists(src_file):
            # Open the image and resize it
            dyn_mask = np.load(src_file)['dyn_mask'][0]
            print(dyn_mask.shape)
            new_size = (512, 288)
            resized_mask = cv2.resize(dyn_mask, new_size, interpolation=cv2.INTER_NEAREST)

            print(resized_mask.shape)
            resized_mask = np.expand_dims(resized_mask, axis=0)

            # Save the resized mask back if needed
            np.savez_compressed(dest_file, dyn_mask=resized_mask)
            # Save the resized image to the destination directory
            # resized_img.save(dest_file)
            print(f'Resized and copied {src_file} to {dest_file}')
        else:
            print(f'{src_file} does not exist')

    print('Copying and resizing process completed.')
