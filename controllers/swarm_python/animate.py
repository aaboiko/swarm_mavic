import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

from tqdm import tqdm

from matplotlib.patches import Circle

x_bound_min = -10
x_bound_max = 10
y_bound_min = -10
y_bound_max = 10
z_bound_min = -1
z_bound_max = 10

x_bounds = (-10, 10)
y_bounds = (-10, 10)
z_bounds = (0, 20)

s_point = 10
s_anchor = 20

data = []
fig, ax = plt.subplots()
x_anchor = np.array([0.0, 0.0, 1.0])
R_vis = 2.0   
traj_len = 4800
n_frames = 2500

family_colors = ["green", "blue", "orange", "black", "brown", "aquamarine", "aqua", "azure", "coral", "chocolate", "purple", "teal", "pink", "gold", "violet", "magenta"]


def animate(i):
    nums = data[i]
    plt.clf()

    plt.xlim((y_bound_min, y_bound_max))
    plt.ylim((z_bound_min, z_bound_max))

    x_anc, y_anc, z_anc = x_anchor
    plt.scatter(x_anc, z_anc, s=s_anchor, color="red")

    for i in np.arange(0, len(nums), 3):
            x = nums[i]
            y = nums[i + 1]
            z = nums[i + 2]

            plt.scatter(x, z, s=10, color='blue')


def animate_1_xy(i):
    with open(traj_path, "r") as traj_file:
        traj_lines = [traj_line for traj_line in traj_file]

    
    print(f"animate: {i}/{len(data)}")
    nums = data[i]
    points = np.array(nums).reshape(-1, 11)
    anchor_pose = [float(item) for item in traj_lines[i % traj_len].rstrip().split(' ')]
    x_anc, y_anc, z_anc = anchor_pose
    circle_colors = ["blue", "green", "red"]
    plt.clf()

    plt.xlim((-5, 5))
    plt.ylim((-5, 5))

    plt.gca().set_aspect('equal', adjustable='box')

    plt.scatter(x_anc, y_anc, s=s_anchor, color="red")

    for point in points:
        x, y, z, dx, dy, dz, ux, uy, uz, peer_state = point
        circle_color = circle_colors[int(peer_state)]

        plt.scatter(x, y, s=s_point, color="blue")
        circle = Circle(xy=(x, y), radius=R_vis, color=circle_color, alpha=0.2)

        plt.gca().add_artist(circle)


def animate_1_xz(i):
    with open(traj_path, "r") as traj_file:
        traj_lines = [traj_line for traj_line in traj_file]

    
    print(f"animate: {i}/{len(data)}")
    nums = data[i]
    points = np.array(nums).reshape(-1, n_params)
    anchor_pose = [float(item) for item in traj_lines[i % traj_len].rstrip().split(' ')]
    x_anc, y_anc, z_anc = anchor_pose
    circle_colors = ["blue", "green", "red"]
    plt.clf()

    plt.xlim((-5, 5))
    plt.ylim((0, 10))

    plt.gca().set_aspect('equal', adjustable='box')

    plt.scatter(x_anc, z_anc, s=s_anchor, color="red")

    for point in points:
        x, y, z, dx, dy, dz, ux, uy, uz, peer_state = point
        circle_color = circle_colors[int(peer_state)]

        plt.scatter(x, z, s=s_point, color="blue")
        circle = Circle(xy=(x, z), radius=R_vis, color=circle_color, alpha=0.2)

        plt.gca().add_artist(circle)


def create_gif(log_path, gif_path):
    iters = 0
    with open(log_path, "r") as file:
        for line in file:
            nums = [float(item) for item in line.rstrip().split(' ')]
            
            data.append(nums)
            iters += 1

            if iters > n_frames:
                 break

    # создаем анимацию
    anim = animation.FuncAnimation(fig, animate_1_xz,  frames = range(0, n_frames, 4), interval = 20)
    # сохраняем анимацию в формате GIF в папку со скриптом
    anim.save(gif_path, fps = 20, writer = 'pillow')


log_path = "logs/point_mass/log_acc_4.txt"
gif_path = "gifs/gif_acc_xz_4.gif"
traj_path = "logs/trajs/circle_xy.txt"
n_params = 12

create_gif(log_path, gif_path)
