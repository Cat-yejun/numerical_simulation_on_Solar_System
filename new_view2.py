import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from datetime import timedelta, datetime
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

Planets = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
dot_color = ['red', 'gray', 'orange', 'blue', 'pink', 'brown', 'yellow', 'cyan', 'blue']

def create_animation_earth_rotation_adjusted(fig, ax, adjusted_phis_all_deg, thetas_all_deg, time_steps, latitude_obs, start_date):
    def update_plot_earth_rotation_adjusted(frame):
        ax.clear()

        # Recreate the semi-circle representing the sky
        sky = patches.Circle((0, 0), 1, color='lightblue', alpha=0.3, clip_on=False)
        ax.add_patch(sky)

        # Set the limits and labels
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Azimuthal Angle (Degrees)')
        ax.set_ylabel('Altitude')

        # Calculate the current time in the simulation
        current_simulation_time = timedelta(seconds=frame * time_step_duration)
        current_datetime = start_date + current_simulation_time
        ax.set_title(f'Southern Sky View from Latitude {latitude_obs}°\nDate & Time: {current_datetime.strftime("%Y-%m-%d %H:%M:%S")}')

        # Plot the positions of the planets at this frame
        for planet in range(thetas_all_deg.shape[0]):
            # Skip Earth since it's the reference point
            if planet == 2:
                continue

            azimuth = adjusted_phis_all_deg[planet, frame]

            # Focus on the southern sky (90° to 270° in right ascension)
            if 90 <= azimuth <= 270:
                # Calculate the altitude and azimuth
                altitude = 90 - np.abs(latitude_obs - thetas_all_deg[planet, frame])
                # Normalize azimuth to plot on the semi-circle
                azimuth_normalized = np.cos(np.radians(azimuth))

                # Plot the planet
                ax.plot(azimuth_normalized, altitude / 90, 'o', color = dot_color[planet], label=f'{Planets[planet]}')

        # Add a legend
        ax.legend(loc='upper right')

    # Create the animation
    ani = animation.FuncAnimation(fig, update_plot_earth_rotation_adjusted, frames=len(time_steps), interval=100)

    return ani

# 이제 함수는 time_step_duration을 인자로 받습니다.
# 함수를 사용할 때는 time_step_duration을 계산하여 전달해야 합니다.



# Function to create an animation for fixed time observation focusing on the southern sky
def create_animation_fixed_time(fig, ax, phis_all_deg, thetas_all_deg, midnight_frames, latitude_obs, start_date):
    def update_plot_fixed_time(frame_idx):
        frame = midnight_frames[frame_idx]
        ax.clear()

        # Recreate the semi-circle representing the sky
        sky = patches.Circle((0, 0), 1, color='lightblue', alpha=0.3, clip_on=False)
        ax.add_patch(sky)

        # Set the limits and labels
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Azimuthal Angle (Degrees)')
        ax.set_ylabel('Altitude')

        # Calculate the current date in the simulation
        current_simulation_date = start_date + timedelta(days=frame_idx)
        ax.set_title(f'Southern Sky View from Latitude {latitude_obs}°\nDate: {current_simulation_date.strftime("%Y-%m-%d")}')

        # Plot the positions of the planets at this frame
        for planet in range(thetas_all_deg.shape[0]):
            # Skip Earth since it's the reference point
            if planet == 3:
                continue

            azimuth = phis_all_deg[planet, frame]

            # Focus on the southern sky (90° to 270° in right ascension)
            if 90 <= azimuth <= 270:
                # Calculate the altitude and azimuth
                altitude = 90 - np.abs(latitude_obs - thetas_all_deg[planet, frame])
                # Normalize azimuth to plot on the semi-circle
                azimuth_normalized = np.cos(np.radians(azimuth))

                # Plot the planet
                ax.plot(azimuth_normalized, altitude / 90, 'o', color = dot_color[planet], label=f'{Planets[planet]}')

        # Add a legend
        ax.legend(loc='upper right')

    # Create the animation
    ani = animation.FuncAnimation(fig, update_plot_fixed_time, frames=len(midnight_frames), interval=100)

    return ani



# Function to convert Cartesian coordinates to spherical coordinates
def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # polar angle
    phi = np.arctan2(y, x)  # azimuthal angle
    return r, theta, phi



def update_view(num, Qs, points_3d, lines_3d, points_xy, lines_xy, points_yz, lines_yz, points_xz, lines_xz, max_x, max_y, max_z, ax):
    num = num*10
    Sun_x_offset = Qs[3][num, 0]
    Sun_y_offset = Qs[3][num, 1]
    Sun_z_offset = Qs[3][num, 2]

    for i in range(len(Qs)):
        Qs[i][:, 0] -= Sun_x_offset 
        Qs[i][:, 1] -= Sun_y_offset 
        Qs[i][:, 2] -= Sun_z_offset
        
        # 3d
        points_3d[i].set_offsets(Qs[i][num, :2])
        points_3d[i].set_3d_properties(Qs[i][num, 2], zdir='z')  # specify zdir
        lines_3d[i].set_data(Qs[i][:num, 0], Qs[i][:num, 1])
        lines_3d[i].set_3d_properties(Qs[i][:num, 2], zdir='z')

        # # xy plane
        # points_xy[i].set_offsets(Qs[i][num, :2])
        # points_xy[i].set_3d_properties(zs=-max_z, zdir='z')  # specify zdir
        # lines_xy[i].set_data(Qs[i][:num, 0], Qs[i][:num, 1])
        # lines_xy[i].set_3d_properties(zs=-max_z, zdir='z')

        # # yz plane
        # points_yz[i].set_offsets(Qs[i][num, 1:])
        # points_yz[i].set_3d_properties(zs=-max_x, zdir='x')  # specify zdir
        # lines_yz[i].set_data(Qs[i][:num, 1], Qs[i][:num, 2])
        # lines_yz[i].set_3d_properties(zs=-max_x, zdir='x')

        # # xz plane
        # points_xz[i].set_offsets(Qs[i][num, ::2])
        # points_xz[i].set_3d_properties(zs=max_y, zdir='y')  # specify zdir
        # lines_xz[i].set_data(Qs[i][:num, 0], Qs[i][:num, 2])
        # lines_xz[i].set_3d_properties(zs=max_y, zdir='y')

    # Earth를 중심으로 plot의 범위를 조정
    center_x = 0  # 이제 Earth가 항상 (0,0)에 있기 때문에 중심 좌표는 0,0입니다.
    center_y = 0
    center_z = 0
    window_size = 5 # 이 값을 조절하여 원하는 범위를 설정할 수 있습니다.
    
    ax.set_xlim(center_x - window_size, center_x + window_size)
    ax.set_ylim(center_y - window_size, center_y + window_size)
    ax.set_zlim(center_z - window_size, center_z + window_size)






# Load the provided data files
realX = np.load('realX.npy')
realY = np.load('realY.npy')
realZ = np.load('realZ.npy')

# Earth is assumed to be at index 2
earthX = realX[3]
earthY = realY[3]
earthZ = realZ[3]


# Calculate the position of each planet relative to Earth
relative_positions = np.array([realX - earthX, realY - earthY, realZ - earthZ])

# Convert relative positions to spherical coordinates
spherical_positions_all = np.zeros_like(relative_positions)
for planet in range(relative_positions.shape[1]):
    for time_step in range(relative_positions.shape[2]):
        x, y, z = relative_positions[:, planet, time_step]
        spherical_positions_all[:, planet, time_step] = cartesian_to_spherical(x, y, z)

# Extracting radial distances, thetas (declinations), and phis (right ascensions)
radial_distances_all, thetas_all, phis_all = spherical_positions_all

# Convert thetas and phis to degrees
thetas_all_deg = np.degrees(thetas_all)
phis_all_deg = np.degrees(phis_all)

# Define the latitude of observation and the start date of the simulation
latitude_obs = 35
start_date = datetime(2023, 10, 27)

# # 함수 사용 예시
# fig, ax = plt.subplots(figsize=(10, 5))
# plt.close(fig)  # 빈 초기 플롯을 표시하지 않기 위함

# Calculate the real time duration of each time step (assuming one year of simulation)
total_simulation_time = 365 * 24 * 60 * 60  # 1 year in seconds
time_steps = np.arange(realX.shape[1])
time_step_duration = total_simulation_time / len(time_steps)  # duration of each time step in seconds

# Calculate the frame indices for fixed time observation (midnight of each day)
frames_per_day = len(time_steps) / 365
midnight_frames = np.round(np.arange(0, len(time_steps), frames_per_day)).astype(int)

# Earth's rotation rate in degrees per hour (360 degrees per 24 hours)
rotation_rate_per_hour = 360 / 24

# Calculate the rotation angle for each time step
rotation_angles = rotation_rate_per_hour * time_steps / (60 * 60)  # Converting time steps to hours

# Adjust the right ascension for each planet at each time step
adjusted_phis_all_deg = (phis_all_deg + rotation_angles[np.newaxis, :]) % 360

# Checking a small sample for verification
adjusted_phis_all_deg_sample = adjusted_phis_all_deg[:, :5]  # Sample first 5 time steps



    
fig2 = plt.figure(figsize=(15, 10))
    
T = []

X = [[0 for j in range(len(realX[0]))] for i in range(len(realX))]
Y = [[0 for j in range(len(realX[0]))] for i in range(len(realX))]
Z = [[0 for j in range(len(realX[0]))] for i in range(len(realX))]


VX = [[] for i in range(len(realX))]
VY = [[] for i in range(len(realX))]
VZ = [[] for i in range(len(realX))]

Q = [[] for i in range(len(realX))]


for i in range(len(realX)):
    X[i] = relative_positions[0][i][::10]
    Y[i] = relative_positions[1][i][::10]
    Z[i] = relative_positions[2][i][::10]

    Q[i] = np.dstack((X[i], Y[i], Z[i]))[0]

N = len(X[0])

min_x, max_x = -5, 5
min_y, max_y = -5, 5
min_z, max_z = -5, 5

# 3D 플롯 초기화
ax1 = fig2.add_subplot(1, 1, 1, projection='3d')
lines_3d = [ax1.plot([], [], [])[0] for _ in Q]
points_3d = [ax1.scatter([], [], []) for _ in Q]

lines_3d_xy = [ax1.plot([], [], zs=5, zdir='z', color='silver')[0] for _ in Q]
points_3d_xy = [ax1.scatter([], [], zs=-max_z, zdir='z', color='silver') for _ in Q]

lines_3d_yz = [ax1.plot([], [], zs=5, zdir='x', color='silver')[0] for _ in Q]
points_3d_yz = [ax1.scatter([], [], zs=-max_x, zdir='x', color='silver') for _ in Q]

lines_3d_xz = [ax1.plot([], [], zs=5, zdir='y', color='silver')[0] for _ in Q]
points_3d_xz = [ax1.scatter([], [], zs=max_y, zdir='y', color='silver') for _ in Q]

ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")

ax1.set_xlim(min_x, max_x)
ax1.set_ylim(min_y, max_y)
ax1.set_zlim(min_z, max_z)

ani1 = animation.FuncAnimation(fig2, update_view, N, fargs=(Q, points_3d, lines_3d,
                                                            points_3d_xy, lines_3d_xy,
                                                            points_3d_yz, lines_3d_yz,
                                                            points_3d_xz, lines_3d_xz,
                                                            max_x, max_y, max_z, ax1), interval=10, blit=False)
# ani1.save('3d.gif', writer='imagemagick', fps=30)


# # XY 평면 정사영 초기화
# ax2 = fig.add_subplot(1, 2, 2)
# lines_xy = [ax2.plot([], [])[0] for _ in Q]
# points_xy = [ax2.scatter([], []) for _ in Q]

# ax2.set_xlabel("x")
# ax2.set_ylabel("y")

# ax2.set_xlim(min_x, max_x)
# ax2.set_ylim(min_y, max_y)

# ax2.set_aspect('equal', 'box')
# ax2.grid()
# ax2.set_title("xy plane (2d view)")

# ani2 = animation.FuncAnimation(fig, update_2d, N, fargs=(Q, points_xy, lines_xy, 0), interval=1, blit=False)
plt.show()

ax2 = fig2.add_subplot(2, 2, 2)




fig_fixed_time, ax_fixed_time = plt.subplots(figsize=(10, 5))
ani_fixed_time = create_animation_fixed_time(fig2, ax2, phis_all_deg, thetas_all_deg, midnight_frames, latitude_obs, start_date)

plt.show()
plt.close(fig_fixed_time)

# Creating animations (functions defined earlier)
fig_earth_rotation_adjusted, ax_earth_rotation_adjusted = plt.subplots(figsize=(10, 5))
ani_earth_rotation_adjusted = create_animation_earth_rotation_adjusted(fig_earth_rotation_adjusted, ax_earth_rotation_adjusted, adjusted_phis_all_deg, thetas_all_deg, time_steps, latitude_obs, start_date)

plt.show()
plt.close(fig_earth_rotation_adjusted)

