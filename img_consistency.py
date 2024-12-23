import os

def rename_images(input_folder, output_folder):
    """
    Renames image files in the input folder sequentially and saves them to the output folder.
    
    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to save renamed images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get all PNG files from the input folder
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith('.png')])
    
    for idx, filename in enumerate(image_files):
        # Create a new filename with consistent naming
        new_filename = f"frame_{idx + 1:04d}.png"
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, new_filename)
        
        # Rename (move) the file
        os.rename(input_path, output_path)
        print(f"Renamed {filename} -> {new_filename}")

    print(f"All images renamed and saved to {output_folder}")

# Example usage
input_folder = "/home/rikhilgupta/Desktop/Surface_Reconstruction/data/statue/images_raw"  # Replace with the path to your images
output_folder = "/home/rikhilgupta/Desktop/Surface_Reconstruction/data/statue/images"  # Replace with the path to save renamed images
rename_images(input_folder, output_folder)
