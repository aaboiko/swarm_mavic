import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

n_params = 13

x_bounds = (-10, 10)
y_bounds = (-10, 10)
z_bounds = (0, 20)

n_frames = 10000
dt = 0.02

fig_3d = plt.figure(figsize = (20, 20))
ax_3d = fig_3d.add_subplot(projection='3d')

traj_path_spinning_line = "grid_antenna/logs/anchors_predefined/spinning_line.txt"
traj_path_sliding_line = "grid_antenna/logs/anchors_predefined/sliding_line.txt"
traj_path_static_sin = "grid_antenna/logs/anchors_predefined/static_sin.txt"

traj_path_circle_xy = "logs/trajs/circle_xy.txt"

traj_agents = "logs/point_mass/log_single_anchor_circle.txt"


def check_traj(traj_path):
    with open(traj_path, "r") as file:
        lines = [line for line in file]
        xs_all, ys_all, zs_all = [], [], []

        for line in tqdm(lines[0:-1:100]):
            nums = [float(item) for item in line.rstrip().split(' ')]
            points = np.array(nums).reshape(-1, n_params)
            xs, ys, zs = [], [], []

            for point in points:
                x, y, z, dx, dy, dz, ux, uy, uz, peer_state, family_id, R_vis, is_alive = point
                xs.append(x)
                ys.append(y)
                zs.append(z)

            xs_all.append(xs)
            ys_all.append(ys)
            zs_all.append(zs)
        
        for i in range(len(xs_all[0])):
            xs_to_plot = np.array(xs_all)[:,i].tolist()
            ys_to_plot = np.array(ys_all)[:,i].tolist()
            zs_to_plot = np.array(zs_all)[:,i].tolist()

            ax_3d.plot(xs_to_plot, ys_to_plot, zs_to_plot, color="blue")

        ax_3d.set_xlim(x_bounds)
        ax_3d.set_ylim(y_bounds)
        ax_3d.set_zlim(z_bounds)

        plt.show()


def check_static_points(traj_path):
    with open(traj_path, "r") as file:
        lines = [line for line in file]

        for line in tqdm(lines[0:-1:100]):
            nums = [float(item) for item in line.rstrip().split(' ')]
            points = np.array(nums).reshape(-1, n_params)

            for point in points:
                x, y, z, dx, dy, dz, ux, uy, uz, peer_state, family_id, R_vis, is_alive = point
                ax_3d.scatter(x, y, z, s=30, color="blue")

        ax_3d.set_xlim(x_bounds)
        ax_3d.set_ylim(y_bounds)
        ax_3d.set_zlim(z_bounds)

        plt.show()


def spinning_line(traj_path, gap, n_agents, steps_from_center=0, rot_angle=np.pi/2):
    open(traj_path, "w").close()

    with open(traj_path, "w") as file:
        for t in tqdm(np.linspace(0, rot_angle, n_frames)):
            line = f""

            for i in range(steps_from_center, n_agents + steps_from_center):
                x = i * gap * np.cos(t)
                y = i * gap * np.sin(t)
                z = 1

                dx = -(1 / (dt * n_frames)) * i * gap * np.sin(t)
                dy = (1 / (dt * n_frames)) * i * gap * np.cos(t)

                line += f"{x} {y} {z} {dx} {dy} {0} {0} {0} {0} {0} {i} {0} {1} "

            file.write(line + "\n")


def static_sin(traj_path, n_agents):
    with open(traj_path, "w") as file:
        for t in tqdm(range(n_frames)):
            x_step = 18 / (n_agents - 1)
            line = f""

            for i in range(n_agents):
                x = i * x_step - 9
                y = 5 * np.sin(2 * np.pi * x / 18)
                z = 1

                line += f"{x} {y} {z} {0} {0} {0} {0} {0} {0} {0} {i} {0} {1} "

            file.write(line + "\n")


def sliding_line(traj_path, n_agents):
    with open(traj_path, "w") as file:
        for t in tqdm(range(n_frames)):
            line = f""

            for i in range(n_agents):
                x = 0.002 * t - 10
                y = i - 5
                z = 1

                line += f"{x} {y} {z} {0.002} {0} {0} {0} {0} {0} {0} {i} {0} {1} "

            file.write(line + "\n")


def chain_initialization(n_agents, R_vis, R_min, sigma_alpha, sigma_beta, x_anchor=0, y_anchor=0, z_anchor=0, show=True):
    points = []
    cur_x, cur_y, cur_z = x_anchor, y_anchor, z_anchor

    for i in range(n_agents):
        interval = np.random.uniform(R_min, R_vis)
        alpha = np.random.normal(0, sigma_alpha**2)
        beta = np.random.normal(0, sigma_beta**2)

        x = cur_x + interval * np.sin(beta)
        y = cur_y - interval * np.sin(alpha) * np.cos(beta)
        z = cur_z + interval * np.cos(alpha) * np.cos(beta)

        points.append(np.array([x, y, z]))

        cur_x, cur_y, cur_z = x, y, z

    if show:
        for point in points:
            x, y, z = point
            ax_3d.scatter(x, y, z, s=30, color="blue")

        ax_3d.set_xlim(x_bounds)
        ax_3d.set_ylim(y_bounds)
        ax_3d.set_zlim(z_bounds)

        plt.show()


def circle_xy(traj_path):
    open(traj_path, "w").close()

    with open(traj_path, "w") as file:
        for t in tqdm(np.linspace(0, np.pi, n_frames)):
            x = 4 * np.cos(t)
            y = 4 * np.sin(t)
            z = 1

            line = f"{x} {y} {z}\n"
            file.write(line)


def static_sin_points(n_agents=13):
    points = []
    x_step = 18 / (n_agents - 1)

    for i in range(n_agents):
        x = i * x_step - 9
        y = 5 * np.sin(2 * np.pi * x / 18)
        z = 1
        points.append((x, y, z))

    return points


def spinning_line_points(n_agents=10, gap=1, steps_from_center=0):
    points = []

    for i in range(steps_from_center, n_agents + steps_from_center):
        x = i * gap
        y = 0
        z = 1
        points.append((x, y, z))

    return points


def sliding_line_points(n_agents=10, gap=1):
    points = []

    for i in range(n_agents):
        x = -10
        y = i * gap - 5
        z = 1
        points.append((x, y, z))

    return points


def chain_poses(n_agents, R_vis, R_min, sigma_alpha, sigma_beta, x_anchor=0, y_anchor=0, z_anchor=0):
    points = []
    cur_x, cur_y, cur_z = x_anchor, y_anchor, z_anchor

    for i in range(n_agents):
        interval = np.random.uniform(R_min, R_vis)
        alpha = np.random.normal(0, sigma_alpha**2)
        beta = np.random.normal(0, sigma_beta**2)

        x = cur_x + interval * np.sin(beta)
        y = cur_y - interval * np.sin(alpha) * np.cos(beta)
        z = cur_z + interval * np.cos(alpha) * np.cos(beta)

        points.append(np.array([x, y, z]))

        cur_x, cur_y, cur_z = x, y, z

    return points


def draw_control(traj_path, id):
    with open(traj_path, "w") as file:
        line_num = 0
        t_prev, vals_prev, alives_prev = 0, [], []

        fig = plt.figure(figsize = (20, 20))
        ax = fig.add_subplot()

        for line in tqdm(file):
            t = line_num * dt

            nums = [float(item) for item in line.rstrip().split(' ')]
            points = np.array(nums).reshape(-1, n_params)
            vals, alives = [], []

            for point in points:
                x, y, z, dx, dy, dz, ux, uy, uz, peer_state, id, is_alive = point


#spinning_line(traj_path_spinning_line, 1, 6, steps_from_center=2, rot_angle=np.pi/3)
#sliding_line(traj_path_sliding_line, 10)

#check_traj(traj_path_spinning_line)
#check_static_points(traj_path_static_sin)

#chain_initialization(50, 1.0, 0.2, 0.1, 0.8)
#circle_xy(traj_path_circle_xy)