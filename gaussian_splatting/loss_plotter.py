import numpy as np
import matplotlib.pyplot as plt

# File path to the NumPy array
file_path = "/home/rikhilgupta/Desktop/Compact_Gaussian_study/gaussian-splatting/fern_loss_list.npy"

# Load the NumPy array
loss_list = np.load(file_path)

# Plot the loss values
plt.figure(figsize=(10, 6))
plt.plot(loss_list, label='Loss')
plt.title("Loss Plot", fontsize=16)
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.legend()
plt.grid(True)

# Save the plot as an image
output_image_path = "/home/rikhilgupta/Desktop/fern_loss_plot.png"
plt.savefig(output_image_path)

print(f"Plot saved as {output_image_path}")
