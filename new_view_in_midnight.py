import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.dates as mdates
from datetime import datetime

dot_color = ['red', 'gray', 'orange', 'blue', 'pink', 'brown', 'yellow', 'cyan', 'blue']

# Load the provided coordinates
x_coords = np.load('realX.npy')
y_coords = np.load('realY.npy')
z_coords = np.load('realZ.npy')

# Check the shape of the loaded data to understand the structure
x_coords.shape, y_coords.shape, z_coords.shape

def project_to_2d(x, y, z, observer_latitude=35.0):
    """
    Project the 3D coordinates onto a 2D plane from the perspective of an observer at a given latitude.
    The observer is assumed to be looking south.
    """
    # Convert latitude to radians
    latitude_rad = np.radians(observer_latitude)

    # Adjust for observer's latitude: Rotate around x-axis
    y_rot = y * np.cos(latitude_rad) + z * np.sin(latitude_rad)
    z_rot = -y * np.sin(latitude_rad) + z * np.cos(latitude_rad)

    # For an observer looking south, we just need to plot y vs. z
    return y_rot, z_rot

# Project the coordinates to 2D
x_2d, y_2d = project_to_2d(x_coords, y_coords, z_coords)

def correct_for_axial_tilt(x, y, z, axial_tilt=23.44):
    """
    Correct the positions of the planets for Earth's axial tilt.
    Axial tilt is assumed to be 23.44 degrees (Earth's axial tilt).
    """
    # Convert axial tilt to radians
    tilt_rad = np.radians(axial_tilt)

    # Correct for axial tilt: Rotate around the y-axis
    x_tilted = x * np.cos(tilt_rad) - z * np.sin(tilt_rad)
    z_tilted = x * np.sin(tilt_rad) + z * np.cos(tilt_rad)

    return x_tilted, y, z_tilted

# Apply the correction for Earth's axial tilt
x_tilted, y_tilted, z_tilted = correct_for_axial_tilt(x_coords, y_coords, z_coords)

# Project the corrected coordinates to 2D from the perspective of an observer looking south
x_2d_corrected, y_2d_corrected = project_to_2d(x_tilted, y_tilted, z_tilted)

# Exclude the Sun from the midnight visualization
# Assuming the first index represents the Sun
x_2d_planets = x_2d_corrected[1:]
y_2d_planets = y_2d_corrected[1:]

# Update the 2D animation function
def update_corrected_graph_2d(num):
    for i, line in enumerate(lines):
        line.set_data(x_2d_planets[i, num], y_2d_planets[i, num])
    title.set_text(f'Day {num+1} at Midnight')
    return lines

# Create a new animation with the corrected data
fig, ax = plt.subplots()

# Initial plot with corrected data
lines = [ax.plot(x_2d_planets[i, 0], y_2d_planets[i, 0], 'o', color = dot_color[i+1])[0] for i in range(x_2d_planets.shape[0])]

# Setting the title
title = ax.set_title('Day 1 at Midnight')

# Setting up the animation
ani_2d_corrected = animation.FuncAnimation(fig, update_corrected_graph_2d, frames=np.arange(0, x_2d_planets.shape[1]),
                                           interval=50, blit=True)

# Display the animation
plt.show()