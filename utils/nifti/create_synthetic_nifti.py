import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D array of shape (100, 100, 100) filled with zeros
data = np.zeros((100, 100, 100))

# Set all the values in the array to 1 (white)
data[:, :, :] = 1

# Set the center 50x50x50 region to 0 (black)
data[25:75, 25:75, 25:75] = 0

# Create a red colormap
cmap = plt.cm.Reds

# Create a new figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Get the x, y, and z indices of the non-zero elements of the data array
x, y, z = data.nonzero()

# Display the non-zero elements as a scatter plot with the red colormap
ax.scatter(x, y, z, c=data[x, y, z], cmap=cmap)

# Set the limits of the x, y, and z axes to show the entire cube
ax.set_xlim(0, data.shape[0])
ax.set_ylim(0, data.shape[1])
ax.set_zlim(0, data.shape[2])

# Set the labels of the x, y, and z axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the figure
plt.show()
