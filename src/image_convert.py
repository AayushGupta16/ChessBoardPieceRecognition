import sys
from pathlib import Path

# Add the yolov5 directory to the system path
yolov5_path = Path.cwd() / 'yolov5'
sys.path.append(str(yolov5_path))

import subprocess

# Set your custom weights and input image path
weights_path = '/Users/aayushgupta/yolov5/runs/train/exp4/weights/best.pt'
input_image_path = '/Users/aayushgupta/yolov5/IMG_0141.JPG'

# Construct the command to run the detect.py script
command = f"python detect.py --weights {weights_path} --source {input_image_path}"

# Run the command using the subprocess module
process = subprocess.run(command, shell=True, check=True)

