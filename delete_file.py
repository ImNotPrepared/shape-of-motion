import os
import glob

# Define the base directory and camera subdirectories
base_dir = "/data3/zihanwa3/Capstone-DSR/shape-of-motion/data/images/"
cameras = ["soccer_undist_cam01", "soccer_undist_cam02", "soccer_undist_cam03", "soccer_undist_cam04"]

# Loop through each camera directory
for cam in cameras:
    cam_dir = os.path.join(base_dir, cam)
    if not os.path.exists(cam_dir):
        print(f"Directory not found: {cam_dir}")
        continue
    
    # Find all JPG files
    for file_path in glob.glob(os.path.join(cam_dir, "*.jpg")):
        file_name = os.path.basename(file_path)
        try:
            # Extract the numeric part of the filename
            file_number = int(file_name.split(".")[0])
            
            # Delete if the number is less than 437
            if file_number < 10070:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        except ValueError:
            print(f"Skipping non-numeric filename: {file_name}")
