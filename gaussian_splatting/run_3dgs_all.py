import subprocess

# List of datasets
# datasets = ['fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex']
datasets = ['statue']#, 'garden', 'kitchen', 'room', 'stump']
# Base paths
base_data_path = "/home/rikhilgupta/Desktop/Surface_Reconstruction/data/"
base_output_path = "/home/rikhilgupta/Desktop/Surface_Reconstruction/gaussian_splatting/output/mit_statue/"

# Common parameters
# train_params = "--eval -r 4 --iterations 30000"
# render_params = "-r 4 --iteration 30000"
train_params = "--eval --iterations 30000"
render_params = "--iteration 30000"


# python train.py -s /home/rikhilgupta/Desktop/Data/mipnerf360/bicycle -m /home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/output/mipnerf360_3dgs_24vews_none_size_thresh/bicycle --eval -r 4 --iterations 30000
# Loop through datasets
for dataset in datasets:
    data_path = f"{base_data_path}{dataset}"
    output_path = f"{base_output_path}{dataset}"
    
    # # Train command
    # train_command = f"python train.py -s {data_path} -m {output_path} {train_params}"
    # print(f"Running: {train_command}")
    # subprocess.run(train_command, shell=True)
    
    # # Render command
    # render_command = f"python render_gurutva.py -m {output_path} {render_params} -s {data_path}"
    # print(f"Running: {render_command}")
    # subprocess.run(render_command, shell=True)
    
    # Video Render command
    video_render_command = f"python render_gurutva.py -s {data_path} -m {output_path} {render_params} --video --skip_train --skip_test"
    print(f"Running: {video_render_command}")
    subprocess.run(video_render_command, shell=True)

    # Metrics command
    # metrics_command = f"python metrics_gurutva.py -m {output_path}"
    # print(f"Running: {metrics_command}")
    # subprocess.run(metrics_command, shell=True)
