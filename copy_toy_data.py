import shutil
import os

# Define source and destination directories

for index in range(1,5):
  src_dir = f'/data3/zihanwa3/Capstone-DSR/Processing/undist_data/undist_cam0{index}/'
  dest_dir = f'/data3/zihanwa3/Capstone-DSR/shape-of-motion/data/images/toy_{index}/'

  # Ensure the destination directory exists
  os.makedirs(dest_dir, exist_ok=True)

  # Define the range of image numbers to copy
  start_index = 183
  end_index = 294

  # Loop through the range and copy each file
  for i in range(start_index, end_index + 1):
      src_file = os.path.join(src_dir, f'{i:05}.jpg')
      dest_file = os.path.join(dest_dir, f'{i:05}.jpg')
      if os.path.exists(src_file):
          shutil.copy(src_file, dest_file)
          print(f'Copied {src_file} to {dest_file}')
      else:
          print(f'{src_file} does not exist')

  print('Copying process completed.')
