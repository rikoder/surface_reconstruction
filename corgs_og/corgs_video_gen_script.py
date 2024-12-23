import os
import subprocess

# List of scenes
# scenes = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
scenes = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"]
# Base command parts
base_command = "python render_corgs_video.py --render_depth --video"
source_path_template = "/home/rikhilgupta/Desktop/Benchmarks/CoR-GS_og/data/mipnerf360/{scene}"
model_path_template = "/home/rikhilgupta/Desktop/Benchmarks/CoR-GS_og/output/mipnerf360_og_12views/{scene}"

# Loop over each scene and execute the command
for scene in scenes:
    source_path = source_path_template.format(scene=scene)
    model_path = model_path_template.format(scene=scene)
    
    # Construct the command
    command = f"{base_command} --source_path {source_path} -m {model_path}"
    
    print(f"Running command: {command}")
    
    # Run the command
    subprocess.run(command, shell=True, check=True)
