import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from point_mass import PointMass, Anchor
from matplotlib.patches import Circle
from tqdm import tqdm
from functools import partial

N_POINTS = 7
scene_name = "scene_3"

log_path = f"logs/point_mass/log_{scene_name}.txt"

gif_path_xy = f"gifs/{scene_name}_xy.gif"
gif_path_xz = f"gifs/{scene_name}_xz.gif"
gif_path_3d = f"gifs/{scene_name}_3d.gif"

graph_path_z = f"gifs/{scene_name}_z.jpg"
graph_path_xy = f"gifs/{scene_name}_xy.jpg"

traj_path_circle_xy = "logs/trajs/circle_xy.txt"
traj_path_circle_skew = "logs/trajs/circle_skew.txt"
traj_path_line = "logs/trajs/line_xy.txt"
traj_path_sin = "logs/trajs/sin_xy.txt"
traj_path_static = "logs/trajs/static.txt"

R_vis = 1.0
w = 0.5
u_max = 0.5
dt = 0.02
traj_len = 10000

n_frames = 10000
data = []

s_point = 10
s_anchor = 30

x_bounds = (-6, 6)
y_bounds = (-6, 6)
z_bounds = (-2, 12)

x_min, x_max = -5, 5
y_min, y_max = -5, 5
z_min, z_max = 4, 10

points_colors = ["magenta", "blue", "violet", "black", "brown", "aquamarine", "aqua", "gold", "coral", "chocolate", "purple", "teal", "pink", "gold", "violet", "magenta"]

fig_3d = plt.figure(figsize = (20, 20))
ax_3d = fig_3d.add_subplot(projection='3d')


def sign(x):
    if x >= 0:
        return 1
    
    return -1


def xi(g):
    return np.tanh(g)


def get_peers(points, focus_point, anchor_pose):
    peers = []
    peer_state = 0

    if np.linalg.norm(anchor_pose - focus_point.pose) <= R_vis:
        peer_state = 2
        vec = anchor_pose - focus_point.pose
        peers.append(vec)

    for point in points:
        d = np.linalg.norm(point.pose - focus_point.pose)

        if d <= R_vis and point.id != focus_point.id and point.is_alive:
            if peer_state == 0:
                peer_state = 1

            peer_state = max(peer_state, point.peer_state)
            vec = point.pose - focus_point.pose
            peers.append(vec)
    
    return peers, peer_state


def get_nearest_distances(peers, anchor_dir):
    min_dist_x_plus, min_dist_y_plus, min_dist_z_plus = np.inf, np.inf, np.inf
    min_dist_x_minus, min_dist_y_minus, min_dist_z_minus = np.inf, np.inf, np.inf
    id_x_plus, id_y_plus, id_z_plus = -1, -1, -1
    id_x_minus, id_y_minus, id_z_minus = -1, -1, -1

    for i in range(len(peers)):
        peer = peers[i]
        vx, vy, vz = peer

        if vx >= 0:
            if vx < min_dist_x_plus:
                min_dist_x_plus = vx
                id_x_plus = i
        else:
            if -vx < min_dist_x_minus:
                min_dist_x_minus = -vx
                id_x_minus = i

        if vy >= 0:
            if vy < min_dist_y_plus:
                min_dist_y_plus = vy
                id_y_plus = i
        else:
            if -vy < min_dist_y_minus:
                min_dist_y_minus = -vy
                id_y_minus = i

        if vz >= 0:
            if vz < min_dist_z_plus:
                min_dist_z_plus = vz
                id_z_plus = i
        else:
            if -vz < min_dist_z_minus:
                min_dist_z_minus = -vz
                id_z_minus = i

    ids = [id_x_plus, id_y_plus, id_z_plus, id_x_minus, id_y_minus, id_z_minus]
    ids_set = set()

    for id in ids:
        ids_set.add(id)

    mins = [min_dist_x_plus, min_dist_y_plus, min_dist_z_plus, min_dist_x_minus, min_dist_y_minus, min_dist_z_minus]
    distances = []

    for i in range(6):
        if ids[i] == -1:
            if anchor_dir[i] < 0:
                if i == 2 or i == 5:
                    distances.append(w)
                else:
                    distances.append(0)
            else:
                distances.append(R_vis)
        else:
            distances.append(mins[i])
    
    return distances


def control_force(distances, point):
    d_x_plus, d_y_plus, d_z_plus, d_x_minus, d_y_minus, d_z_minus = distances
    vx, vy, vz = point.dpose

    sigma_x = vx - xi(d_x_plus) + xi(d_x_minus)
    sigma_y = vy - xi(d_y_plus) + xi(d_y_minus)
    sigma_z = vz - xi(d_z_plus) + xi(d_z_minus)

    ux = -u_max * sign(sigma_x)
    uy = -u_max * sign(sigma_y)
    uz = -u_max * sign(sigma_z)

    return np.array([ux, uy, uz])


def write_log(path, points):
    with open(path, "a") as file:
        line = f""

        for point in points:
            x, y, z = point.pose
            dx, dy, dz = point.dpose
            ux, uy, uz = point.force
            peer_state = int(point.peer_state)
            id = point.id
            is_alive = int(point.is_alive)

            line += f"{x} {y} {z} {dx} {dy} {dz} {ux} {uy} {uz} {peer_state} {id} {is_alive} "

        file.write(line + "\n")
        #print(line)


def process(anchor_traj_path="logs/trajs/static.txt"):
    '''points = [
        PointMass(id=0, x=-2, y=-2, z=int(np.random.uniform(2, 11))),
        PointMass(id=1, x=-2, y=2, z=int(np.random.uniform(2, 11))),
        PointMass(id=2, x=2, y=-2, z=int(np.random.uniform(2, 11))),
        PointMass(id=3, x=2, y=2, z=int(np.random.uniform(2, 11)))
    ]'''

    points = [PointMass(id=i, x=np.random.uniform(x_min, x_max), y=np.random.uniform(y_min, y_max), z=np.random.uniform(z_min, z_max)) for i in range(N_POINTS)]

    points.append(PointMass(id=N_POINTS, x=20, y=20, z=5))
    points.append(PointMass(id=N_POINTS+1, x=20, y=20, z=7))
    points.append(PointMass(id=N_POINTS+2, x=20, y=20, z=10))

    anchor = Anchor(np.array([0, 0, 1]))
    anchor_traj = []

    with open(anchor_traj_path, "r") as file:
        lines = [line for line in file]

        for line in lines:
            nums = [float(item) for item in line.rstrip().split(' ')]
            x, y, z = nums
            p = np.array([x, y, z])
            anchor_traj.append(p)

    anchor.set_trajectory(anchor_traj)

    n_iters = 10000
    flag = False

    for iter in tqdm(range(n_iters)):
        if iter == 2500:
            flag = True

        write_log(log_path, points)

        for point in points:
            '''if flag and (point.id == 0 or point.id == 3 or point.id == 7):
                point.kill()'''
            
            '''if flag and (point.id == N_POINTS or point.id == N_POINTS + 1 or point.id == N_POINTS + 2):
                point.resurrect()'''

            if point.is_alive:
                peers, p_state = get_peers(points, point, anchor.pose)
                point.peer_state = p_state

                vec_to_anchor = anchor.pose - point.pose
                x_to_anchor, y_to_anchor, z_to_anchor = vec_to_anchor
                point.prior_knowledge = np.array([sign(x_to_anchor), sign(y_to_anchor), sign(z_to_anchor)])
                signs = [sign(x_to_anchor), sign(y_to_anchor), sign(z_to_anchor), -sign(x_to_anchor), -sign(y_to_anchor), -sign(z_to_anchor)]

                if len(peers) > 0:
                    distances = get_nearest_distances(peers, signs)
                else:
                    distances = []

                    for i, item in enumerate(signs):
                        if item >= 0:
                            distances.append(R_vis * item)
                        else:
                            if i == 2 or i == 5:
                                distances.append(-w * item)
                            else:
                                distances.append(0)

                force = control_force(distances, point)
                point.apply_force(force)
                point.step(dt)

        anchor.step()


def animate_from_log(log_path, traj_path, axes="xy"):
    with open(log_path, "r") as file, open(traj_path, "r") as traj_file:
        line_num = 0
        traj_lines = [traj_line for traj_line in traj_file]

        for line in file:
            line_num += 1
            print(f"line number: {line_num}")

            nums = [float(item) for item in line.rstrip().split(' ')]
            points = np.array(nums).reshape(-1, 12)
            anchor_pose = [float(item) for item in traj_lines[line_num % traj_len].rstrip().split(' ')]
            x_anc, y_anc, z_anc = anchor_pose
            circle_colors = ["blue", "green", "red"]
            plt.clf()

            plt.xlabel('x')

            if axes == "xy":
                plt.xlim(x_bounds)
                plt.ylim(y_bounds)
                plt.ylabel('y')
            if axes == "xz":
                plt.xlim(x_bounds)
                plt.ylim(z_bounds)
                plt.ylabel('z')

            plt.gca().set_aspect('equal', adjustable='box')

            if axes == "xy":
                plt.scatter(x_anc, y_anc, s=s_anchor, color="red")
            if axes == "xz":
                plt.scatter(x_anc, z_anc, s=s_anchor, color="red")

            for point in points:
                x, y, z, dx, dy, dz, ux, uy, uz, peer_state, id, is_alive = point
                
                if int(is_alive) > 0:
                    circle_color = circle_colors[int(peer_state)]
                    point_color = points_colors[int(id)]

                    if axes == "xy":
                        plt.scatter(x, y, s=s_point, color=point_color)
                        circle = Circle(xy=(x, y), radius=R_vis, color=circle_color, alpha=0.2)
                    if axes == "xz":
                        plt.scatter(x, z, s=s_point, color=point_color)
                        circle = Circle(xy=(x, z), radius=R_vis, color=circle_color, alpha=0.2)

                    plt.gca().add_artist(circle)

            plt.pause(0.01)

        plt.show()


def animate_xy(i, traj_path=""):
    with open(traj_path, "r") as traj_file:
        traj_lines = [traj_line for traj_line in traj_file]

    nums = data[i]
    points = np.array(nums).reshape(-1, 12)
    anchor_pose = [float(item) for item in traj_lines[i % traj_len].rstrip().split(' ')]
    x_anc, y_anc, z_anc = anchor_pose
    circle_colors = ["blue", "green", "red"]
    plt.clf()

    plt.xlabel('x')
    plt.xlim(x_bounds)
    plt.ylim(y_bounds)
    plt.ylabel('y')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(x_anc, y_anc, s=s_anchor, color="red")

    for point in points:
        x, y, z, dx, dy, dz, ux, uy, uz, peer_state, id, is_alive = point
        
        if int(is_alive) > 0:
            circle_color = circle_colors[int(peer_state)]
            point_color = points_colors[int(id)]

            plt.scatter(x, y, s=s_point, color=point_color)
            circle = Circle(xy=(x, y), radius=R_vis, color=circle_color, alpha=0.2)

            plt.gca().add_artist(circle)


def animate_xz(i, traj_path=""):
    with open(traj_path, "r") as traj_file:
        traj_lines = [traj_line for traj_line in traj_file]

    nums = data[i]
    points = np.array(nums).reshape(-1, 12)
    anchor_pose = [float(item) for item in traj_lines[i % traj_len].rstrip().split(' ')]
    x_anc, y_anc, z_anc = anchor_pose
    circle_colors = ["blue", "green", "red"]
    plt.clf()

    plt.xlabel('x')
    plt.xlim(x_bounds)
    plt.ylim(z_bounds)
    plt.ylabel('z')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(x_anc, z_anc, s=s_anchor, color="red")

    for point in points:
        x, y, z, dx, dy, dz, ux, uy, uz, peer_state, id, is_alive = point
        
        if int(is_alive) > 0:
            circle_color = circle_colors[int(peer_state)]
            point_color = points_colors[int(id)]

            plt.scatter(x, z, s=s_point, color=point_color)
            circle = Circle(xy=(x, z), radius=R_vis, color=circle_color, alpha=0.2)

            plt.gca().add_artist(circle)


def animate_3d(i, traj_path=""):
    with open(traj_path, "r") as traj_file:
        traj_lines = [traj_line for traj_line in traj_file]

    ax_3d.clear()

    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('z')

    ax_3d.set_xlim(x_bounds)
    ax_3d.set_ylim(y_bounds)
    ax_3d.set_zlim(z_bounds)

    ax_3d.set_aspect('equal', adjustable='box')
    circle_colors = ["blue", "green", "red"]
    nums = data[i]

    agents = np.array(nums).reshape(-1, 12)
    anchor_pose = [float(item) for item in traj_lines[i % traj_len].rstrip().split(' ')]
    x_anc, y_anc, z_anc = anchor_pose
    ax_3d.scatter(x_anc, y_anc, z_anc, s=s_anchor, color="red")

    for agent in agents:
        x, y, z, dx, dy, dz, ux, uy, uz, peer_state, id, is_alive = agent
        
        if int(is_alive) > 0:
            circle_color = circle_colors[int(peer_state)]
            point_color = points_colors[int(id)]

            ax_3d.scatter(x, y, z, s=s_point, color=point_color)
            #ax_3d.scatter(x, y, z, s=np.pi*R_vis**2, color=circle_color, alpha=0.2)


def create_gif(log_path, gif_path, traj_path, flag="xy", frame_step=4):
    iters = 0

    with open(log_path, "r") as file:
        lines = [line for line in file]

        for line in lines:
            nums = [float(item) for item in line.rstrip().split(' ')]
            data.append(nums)
            iters += 1

            if iters >= n_frames:
                break
        
        if flag != "3d":
            fig, ax = plt.subplots()

        if flag == "xy":
            anim = animation.FuncAnimation(fig, partial(animate_xy, traj_path=traj_path), frames = tqdm(range(0, n_frames, frame_step)), interval = 20)
        if flag == "xz":
            anim = animation.FuncAnimation(fig, partial(animate_xz, traj_path=traj_path), frames = tqdm(range(0, n_frames, frame_step)), interval = 20)
        if flag == "3d":
            anim = animation.FuncAnimation(fig_3d, partial(animate_3d, traj_path=traj_path), frames = tqdm(range(0, n_frames, frame_step)), interval = 20)

        anim.save(gif_path, fps = 60, writer = 'pillow')


def plot_graph_z(log_path, traj_path, graph_path):
    with open(log_path, "r") as file, open(traj_path, "r") as traj_file:
        line_num = 0
        traj_lines = [traj_line for traj_line in traj_file]
        ts, zs, alives = [], [], []

        for line in tqdm(file):
            t = line_num * dt
            ts.append(t)

            nums = [float(item) for item in line.rstrip().split(' ')]
            points = np.array(nums).reshape(-1, 12)
            anchor_pose = [float(item) for item in traj_lines[line_num % traj_len].rstrip().split(' ')]
            x_anc, y_anc, z_anc = anchor_pose
            circle_colors = ["blue", "green", "red"]
            line_num += 1

            plt.clf()
            pzs = []

            for point in points:
                x, y, z, dx, dy, dz, ux, uy, uz, peer_state, id, is_alive = point
                z_diff = z - z_anc
                pzs.append(z_diff)

            zs.append(pzs)

        for i in range(N_POINTS):
            plt.plot(ts, np.array(zs)[:,i].tolist(), color=points_colors[i])

        plt.xlabel('t', fontsize=14)
        plt.ylabel(r'$z_i - z_{anchor}$', fontsize=14)
        plt.title(r'$z_i - z_{anchor}$ for all agents', fontsize=14)
        plt.ylim(z_bounds)
        
        plt.savefig(graph_path)
        plt.show()


def plot_graph_xy(log_path, traj_path, graph_path):
    with open(log_path, "r") as file, open(traj_path, "r") as traj_file:
        line_num = 0
        traj_lines = [traj_line for traj_line in traj_file]
        ts, ds = [], []

        for line in tqdm(file):
            t = line_num * dt
            ts.append(t)

            nums = [float(item) for item in line.rstrip().split(' ')]
            points = np.array(nums).reshape(-1, 12)
            anchor_pose = [float(item) for item in traj_lines[line_num % traj_len].rstrip().split(' ')]
            x_anc, y_anc, z_anc = anchor_pose
            circle_colors = ["blue", "green", "red"]
            line_num += 1

            plt.clf()
            pds = []

            for point in points:
                x, y, z, dx, dy, dz, ux, uy, uz, peer_state, id, is_alive = point
                x_diff = x - x_anc
                y_diff = y - y_anc
                z_diff = np.linalg.norm(np.array([x_diff, y_diff]))
                pds.append(int(is_alive) * z_diff)

            ds.append(pds)

        for i in range(N_POINTS):
            plt.plot(ts, np.array(ds)[:,i].tolist(), color=points_colors[i])

        plt.xlabel('t', fontsize=14)
        plt.ylabel(r'$d_i$ from anchor Z-axis', fontsize=14)
        plt.title(r'XY-distances from anchor axis for all agents', fontsize=14)
        plt.ylim((-1, 6))
        
        plt.savefig(graph_path)
        plt.show()


def check_traj(traj_path):
    with open(traj_path, "r") as file:
        lines = [line for line in file]

        for line in tqdm(lines[0:-1:100]):
            nums = [float(item) for item in line.rstrip().split(' ')]
            x, y, z = nums
            ax_3d.scatter(x, y, z, s=s_point, color="blue")

        ax_3d.set_xlim(x_bounds)
        ax_3d.set_ylim(y_bounds)
        ax_3d.set_zlim(z_bounds)

        plt.show()


process(anchor_traj_path=traj_path_circle_skew)
#animate_from_log(log_path, traj_path_static, axes="xy")

create_gif(log_path, gif_path_xy, traj_path_circle_skew, flag="xy", frame_step=20)
create_gif(log_path, gif_path_xz, traj_path_circle_skew, flag="xz", frame_step=20)
create_gif(log_path, gif_path_3d, traj_path_circle_skew, flag="3d", frame_step=20)

plot_graph_z(log_path, traj_path_circle_skew, graph_path_z)
plot_graph_xy(log_path, traj_path_circle_skew, graph_path_xy)