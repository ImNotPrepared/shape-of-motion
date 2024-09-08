import os
import shutil
from PIL import Image

for index in range(1, 5):
    src_dir = f'/data3/zihanwa3/Capstone-DSR/shape-of-motion/data/images/toy_{index}/'
    dest_dir = f'/data3/zihanwa3/Capstone-DSR/shape-of-motion/data/images/toy_512_{index}/'

    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Define the range of image numbers to copy and resize
    start_index = 183
    end_index = 294

    # Loop through the range and copy each file, then resize
    for i in range(start_index, end_index + 1):
        src_file = os.path.join(src_dir, f'{i:05}.jpg')
        dest_file = os.path.join(dest_dir, f'{i:05}.jpg')
        
        if os.path.exists(src_file):
            # Open the image and resize it
            img = Image.open(src_file)
            resized_img = img.resize((512, 288))
            
            # Save the resized image to the destination directory
            resized_img.save(dest_file)
            print(f'Resized and copied {src_file} to {dest_file}')
        else:
            print(f'{src_file} does not exist')

    print('Copying and resizing process completed.')
