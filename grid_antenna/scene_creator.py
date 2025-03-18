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


def spinning_line(traj_path, gap, n_agents):
    with open(traj_path, "w") as file:
        for t in tqdm(np.linspace(0, 2 * np.pi, n_frames)):
            line = f""

            for i in range(n_agents):
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


#spinning_line(traj_path_spinning_line, 1, 10)
#static_sin(traj_path_static_sin, 13)
sliding_line(traj_path_sliding_line, 10)

check_traj(traj_path_sliding_line)
#check_static_points(traj_path_static_sin)