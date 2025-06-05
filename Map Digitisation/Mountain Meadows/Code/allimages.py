import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
'''
Join hole layouts into 1
'''

# Set your folder path where hole images are saved
folder = "/Users/federicadomecq/Desktop/Golf ModeL/Map Digitisation/"
output_file = "course_all_holes_yardagealigned.png"

# Choose which holes to include (skip 0 if it exists)
hole_nums = [i for i in range(1, 19)]  # hole 1 through 18

# Load images
images = [mpimg.imread(os.path.join(folder, f"/Users/federicadomecq/Desktop/Golf ModeL/Map Digitisation/Mountain Meadows/Mountain Meadows Images and Layouts/YardageAligned/hole_{i}_yards.png")) for i in hole_nums]

# Grid size (adjust as needed)
cols = 6
rows = -(-len(images) // cols)  # ceiling division

# Create the figure
fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

for i, ax in enumerate(axes.flat):
    if i < len(images):
        ax.imshow(images[i])
        ax.set_title(f"Hole {hole_nums[i]}")
        ax.axis('off')
    else:
        ax.axis('off')  # hide unused axes

plt.tight_layout()
plt.savefig(os.path.join(folder, output_file), dpi=300)
plt.close()
