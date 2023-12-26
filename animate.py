# load file and animate
import N_body_many_planets as Nbody
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
from scipy.signal import find_peaks
from scipy import interpolate
from astropy.coordinates import SkyCoord
import astropy.units as u

Planets = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']


def update_view(num, Qs, points_3d, lines_3d, points_xy, lines_xy, points_yz, lines_yz, points_xz, lines_xz, max_x, max_y, max_z, ax):
    # Sun_x_offset = Qs[0][num, 0]
    # Sun_y_offset = Qs[0][num, 1]
    # Sun_z_offset = Qs[0][num, 2]
    Sun_x_offset = 0
    Sun_y_offset = 0
    Sun_z_offset = 0

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



def update_3d_general(num, Qs, points_3d, lines_3d):
    for i in range(len(Qs)):
        points_3d[i].set_offsets(Qs[i][num, :2])
        points_3d[i].set_3d_properties(Qs[i][num, 2], zdir='z')  # specify zdir
        lines_3d[i].set_data(Qs[i][:num, 0], Qs[i][:num, 1])
        lines_3d[i].set_3d_properties(Qs[i][:num, 2], zdir='z')

def update_3d(num, Qs, points_3d, lines_3d, points_xy, lines_xy, points_yz, lines_yz, points_xz, lines_xz, max_x, max_y, max_z):
    for i in range(len(Qs)):
        # 3d
        points_3d[i].set_offsets(Qs[i][num, :2])
        points_3d[i].set_3d_properties(Qs[i][num, 2], zdir='z')  # specify zdir
        lines_3d[i].set_data(Qs[i][:num, 0], Qs[i][:num, 1])
        lines_3d[i].set_3d_properties(Qs[i][:num, 2], zdir='z')

        # xy plane
        points_xy[i].set_offsets(Qs[i][num, :2])
        points_xy[i].set_3d_properties(zs=-max_z, zdir='z')  # specify zdir
        lines_xy[i].set_data(Qs[i][:num, 0], Qs[i][:num, 1])
        lines_xy[i].set_3d_properties(zs=-max_z, zdir='z')

        # yz plane
        points_yz[i].set_offsets(Qs[i][num, 1:])
        points_yz[i].set_3d_properties(zs=-max_x, zdir='x')  # specify zdir
        lines_yz[i].set_data(Qs[i][:num, 1], Qs[i][:num, 2])
        lines_yz[i].set_3d_properties(zs=-max_x, zdir='x')

        # xz plane
        points_xz[i].set_offsets(Qs[i][num, ::2])
        points_xz[i].set_3d_properties(zs=max_y, zdir='y')  # specify zdir
        lines_xz[i].set_data(Qs[i][:num, 0], Qs[i][:num, 2])
        lines_xz[i].set_3d_properties(zs=max_y, zdir='y')


def update_2d(num, Qs, points, lines, plane):
    for i in range(len(Qs)):
        if(plane==0): # xy plane
            points[i].set_offsets(Qs[i][num, :2])
            lines[i].set_data(Qs[i][:num, 0], Qs[i][:num, 1])

        elif(plane==1): # yz plane
            points[i].set_offsets(Qs[i][num, 1:])
            lines[i].set_data(Qs[i][:num, 1], Qs[i][:num, 2])

        elif(plane==2): # xz plane
            points[i].set_offsets(Qs[i][num, ::2])
            lines[i].set_data(Qs[i][:num, 0], Qs[i][:num, 2])








fig = plt.figure(figsize=(15, 10))

initialTime = Nbody.ts
finalTime = Nbody.ts + 1
interval = 0.0001

Numbers = int((finalTime - initialTime) / interval)
particle = Nbody.particle

# load data


T = []

X = [[0 for j in range(Numbers)] for i in range(len(particle))]
Y = [[0 for j in range(Numbers)] for i in range(len(particle))]
Z = [[0 for j in range(Numbers)] for i in range(len(particle))]

Q = [[] for i in range(len(particle))]

realX = np.load('solar_system/realX.npy')
realY = np.load('solar_system/realY.npy')
realZ = np.load('solar_system/realZ.npy')


# R=[]

# R.append(np.sqrt((realX[0]-realX[1])**2 + (realY[0]-realY[1])**2 + (realZ[0]-realZ[1])**2))
# R.append(np.sqrt((realX[0]-realX[2])**2 + (realY[0]-realY[2])**2 + (realZ[0]-realZ[2])**2))
# R.append(np.sqrt((realX[0]-realX[3])**2 + (realY[0]-realY[3])**2 + (realZ[0]-realZ[3])**2))
# R.append(np.sqrt((realX[0]-realX[4])**2 + (realY[0]-realY[4])**2 + (realZ[0]-realZ[4])**2))

# time = [i for i in range(len(R[0])) ]

# interpolation_function1 = interpolate.interp1d(time, R[0], kind='cubic')
# interpolation_function2 = interpolate.interp1d(time, R[1], kind='cubic')
# interpolation_function3 = interpolate.interp1d(time, R[2], kind='cubic')
# interpolation_function4 = interpolate.interp1d(time, R[3], kind='cubic')

# fine_time_array = np.linspace(time[0], time[-1], num=1000000)

# totalTime = len(R[0])*interval

# interval = totalTime / 1000000

# NR = []

# NR.append(interpolation_function1(fine_time_array))
# NR.append(interpolation_function2(fine_time_array))
# NR.append(interpolation_function3(fine_time_array))
# NR.append(interpolation_function4(fine_time_array))

# # print(len(time))
# # print(len(R[0]))

# # plt.plot(time, R[0], 'red', 
# #          time, R[1], 'orange', 
# #          time, R[2], 'yellow', 
# #          time, R[3], 'green')

# # plt.plot(fine_time_array, NR[0], 'red', 
# #          fine_time_array, NR[1], 'orange', 
# #          fine_time_array, NR[2], 'blue', 
# #          fine_time_array, NR[3], 'green')

# tz = (365*24*60*60)

# peaks1, _ = find_peaks(-NR[0])
# peaks2, _ = find_peaks(-NR[1])
# peaks3, _ = find_peaks(-NR[2])
# peaks4, _ = find_peaks(-NR[3])

# periods1 = np.diff(peaks1)
# periods2 = np.diff(peaks2)
# periods3 = np.diff(peaks3)
# #print(peaks4)
# periods4 = np.diff(peaks4)

# #print(periods1, periods2, periods3, periods4)

# periods_mercury = periods1[0]
# periods_venus = periods2[0]
# periods_earth = periods3[0]
# periods_mars = periods4[0]

# periods_mercury = (periods_mercury*interval*tz)/(24*60*60)
# periods_venus = (periods_venus*interval*tz)/(24*60*60)
# periods_earth = (periods_earth*interval*tz)/(24*60*60)
# periods_mars = (periods_mars*interval*tz)/(24*60*60)

# print(periods_mercury, periods_venus, periods_earth, periods_mars)

frameRate = 100

for i in range(len(particle)):
    X[i] = realX[i][::frameRate]
    Y[i] = realY[i][::frameRate]
    Z[i] = realZ[i][::frameRate]

    Q[i] = np.dstack((X[i], Y[i], Z[i]))[0]

N = len(X[0])


#xmax = max([np.abs(X).max(), np.abs(Y).max(), np.abs(Z).max()])
#ax.set_xlim(-xmax-1, xmax+1)
#ax.set_ylim(-xmax-1, xmax+1)
'''
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_zlim(-1.2, 1.2)

point = [[] for i in range(len(particle))]
line = [[] for i in range(len(particle))]

for i in range(len(X)):
    point[i] = ax.scatter([], [], [], s=50)
    line[i], = ax.plot([], [], [])    
'''
# 원래는 1.2
min_x, max_x = -2, 2
min_y, max_y = -2, 2
min_z, max_z = -2, 2

# 3D 플롯 초기화
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
lines_3d = [ax1.plot([], [], [])[0] for _ in Q]
points_3d = [ax1.scatter([], [], []) for _ in Q]

lines_3d_xy = [ax1.plot([], [], zs=1.2, zdir='z', color='silver')[0] for _ in Q]
points_3d_xy = [ax1.scatter([], [], zs=-max_z, zdir='z', color='silver') for _ in Q]

lines_3d_yz = [ax1.plot([], [], zs=1.2, zdir='x', color='silver')[0] for _ in Q]
points_3d_yz = [ax1.scatter([], [], zs=-max_x, zdir='x', color='silver') for _ in Q]

lines_3d_xz = [ax1.plot([], [], zs=1.2, zdir='y', color='silver')[0] for _ in Q]
points_3d_xz = [ax1.scatter([], [], zs=max_y, zdir='y', color='silver') for _ in Q]

ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")

ax1.set_xlim(min_x, max_x)
ax1.set_ylim(min_y, max_y)
ax1.set_zlim(min_z, max_z)

ani1 = animation.FuncAnimation(fig, update_view, N, fargs=(Q, points_3d, lines_3d,
                                                            points_3d_xy, lines_3d_xy,
                                                            points_3d_yz, lines_3d_yz,
                                                            points_3d_xz, lines_3d_xz,
                                                            max_x, max_y, max_z, ax1), interval=1, blit=False)

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

'''
# YZ 평면 정사영 초기화
ax3 = fig.add_subplot(3, 3, 8)
lines_yz = [ax3.plot([], [])[0] for _ in Q]
points_yz = [ax3.scatter([], []) for _ in Q]

ax3.set_xlabel("y")
ax3.set_ylabel("z")

ax3.set_xlim(min_y, max_y)
ax3.set_ylim(min_z, max_z)

ax3.set_aspect('equal', 'box')
ax3.grid()
ax3.set_title("yz plane (2d view)")

ani3 = animation.FuncAnimation(fig, update_2d, N, fargs=(Q, points_yz, lines_yz, 1), interval=1, blit=False)

# XZ 평면 정사영 초기화
ax4 = fig.add_subplot(3, 3, 9)
lines_xz = [ax4.plot([], [])[0] for _ in Q]
points_xz = [ax4.scatter([], []) for _ in Q]

ax4.set_xlabel("x")
ax4.set_ylabel("z")

ax4.set_xlim(min_x, max_x)
ax4.set_ylim(min_z, max_z)

ax4.set_aspect('equal', 'box')
ax4.grid()
ax4.set_title("xz plane (2d view)")

ani4 = animation.FuncAnimation(fig, update_2d, N, fargs=(Q, points_xz, lines_xz, 2), interval=1, blit=False)

# 3D view from xy plane
ax5 = fig.add_subplot(3, 3, 4, projection='3d')
lines_3d_view_xy = [ax5.plot([], [], [])[0] for _ in Q]
points_3d_view_xy = [ax5.scatter([], [], []) for _ in Q]

ax5.view_init(90, 270)
ax5.set_title("xy plane (3d view)", y=0.95)

ax5.set_xlabel("x")
ax5.set_ylabel("y")

ax5.set_xlim(min_x, max_x)
ax5.set_ylim(min_y, max_y)
ax5.set_zlim(min_z, max_z)

ax5.w_zaxis.set_ticklabels([])

ani5 = animation.FuncAnimation(fig, update_3d_general, N, fargs=(Q, points_3d_view_xy, lines_3d_view_xy), interval=1, blit=False)


# 3D view from yz plane
ax6 = fig.add_subplot(3, 3, 5, projection='3d')
lines_3d_view_yz = [ax6.plot([], [], [])[0] for _ in Q]
points_3d_view_yz = [ax6.scatter([], [], []) for _ in Q]

ax6.view_init(0, 0)
ax6.set_title("yz plane (3d view)", y=0.88)

ax6.set_ylabel("y")
ax6.set_zlabel("z")

ax6.set_xlim(min_x, max_x)
ax6.set_ylim(min_y, max_y)
ax6.set_zlim(min_z, max_z)

ax6.w_xaxis.set_ticklabels([])

ani6 = animation.FuncAnimation(fig, update_3d_general, N, fargs=(Q, points_3d_view_yz, lines_3d_view_yz), interval=1, blit=False)

# 3D view from xz plane
ax7 = fig.add_subplot(3, 3, 6, projection='3d')
lines_3d_view_xz = [ax7.plot([], [], [])[0] for _ in Q]
points_3d_view_xz = [ax7.scatter([], [], []) for _ in Q]

ax7.view_init(0, 270)
ax7.set_title("xz plane (3d view)", y=0.88)

ax7.set_xlabel("x")
ax7.set_zlabel("z")

ax7.set_xlim(min_x, max_x)
ax7.set_ylim(min_y, max_y)
ax7.set_zlim(min_z, max_z)

ax7.w_yaxis.set_ticklabels([])

ani7 = animation.FuncAnimation(fig, update_3d_general, N, fargs=(Q, points_3d_view_xz, lines_3d_view_xz), interval=1, blit=False)
'''

ani1.save('3d.gif', writer='pillow', fps=30)
#ani2.save('xy.gif', writer='pillow', fps=30)

plt.show()
