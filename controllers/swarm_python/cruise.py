import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from point_mass import Agent_1D, Pacemaker
from matplotlib.patches import Circle
from tqdm import tqdm
from functools import partial

n_params = 6

R_vis = 1.0
w = 0.5
u_max = 0.5
dt = 0.02
traj_len = 10000

n_frames = 10000
data = []

s_point = 30
s_anchor = 60

colorize_perception_area = True
points_colors = ["magenta", "blue", "violet", "black", "brown", "aquamarine", "aqua", "gold", "coral", "chocolate", "purple", "teal", "pink", "gold", "violet", "magenta"]
circle_colors = ["blue", "green", "red"]

pacemaker_velocity = -0.0
pacemaker_acc = -0.25
pacemaker_motion_func = lambda t: pacemaker_velocity * t + 0.5 * pacemaker_acc * t**2


def sign(x):
    if x >= 0:
        return 1
    
    return -1


def xi(g):
    return np.tanh(g)


def get_peers(points, focus_point, anchor_pose):
    peers = []
    peer_state = 0
    has_peer_forward = False

    if abs(anchor_pose - focus_point.pose) <= R_vis:
        peer_state = 2
        vec = anchor_pose - focus_point.pose
        peers.append(vec)
        has_peer_forward = True

    for point in points:
        d = abs(point.pose - focus_point.pose) + np.random.normal(0, 0.0001)

    for point in points:
        d = np.linalg.norm(point.pose - focus_point.pose) + np.random.normal(0, 0.0001)

        if d <= R_vis and point.id != focus_point.id and point.is_alive:
            if peer_state == 0:
                peer_state = 1

            peer_state = max(peer_state, point.peer_state)
            vec = point.pose - focus_point.pose
            peers.append(vec)

            if point.pose >= focus_point.pose:
                has_peer_forward = True
    
    return peers, peer_state, has_peer_forward


def get_nearest_distances(peers):
    min_dist_plus, min_dist_minus = np.inf, np.inf
    id_plus, id_minus = -1, -1

    for i in range(len(peers)):
        v = peers[i]

        if v >= 0:
            if v < min_dist_plus:
                min_dist_plus = v
                id_plus = i
        else:
            if -v < min_dist_minus:
                min_dist_minus = -v
                id_minus = i

    has_peer_forward = id_plus > -1
    has_peer_back = id_minus > -1

    return min_dist_plus, min_dist_minus, has_peer_forward, has_peer_back


def g(min_dist, gap, has_peer):
    return min_dist if has_peer else gap


def g_gothic(min_dist, has_peer):
    return min(g(min_dist, R_vis, has_peer), w)


def control_force(min_dist_plus, min_dist_minus, has_peer_forward, has_peer_back, point):
    dx = point.dpose
    sigma = dx - xi(g_gothic(min_dist_plus, has_peer_forward)) + xi(g(min_dist_minus, R_vis, has_peer_back))
    u = -u_max * sign(sigma)

    return u


def write_log(path, points):
    with open(path, "a") as file:
        line = f""

        for point in points:
            x = point.pose
            dx = point.dpose
            u = point.force
            peer_state = int(point.peer_state)
            id = point.id
            is_alive = int(point.is_alive)

            line += f"{x} {dx} {u} {peer_state} {id} {is_alive} "

        file.write(line + "\n")


def process(n_points,
            log_path,
            ):
    
    print('processing...')

    R_min = 0.1
    points = []
    cur_x = 0

    for i in range(n_points):
        interval = np.random.uniform(R_vis, 2*R_vis)
        x = cur_x + interval
        point = Agent_1D(id=i, x=x)
        points.append(point)
        cur_x = x

    pacemaker = Pacemaker(pacemaker_motion_func)

    n_iters = 10000
    open(log_path, "w").close()

    for iter in tqdm(range(n_iters)):
        write_log(log_path, points)

        for point in points:
            if point.is_alive:
                peers, p_state, has_peer_forward = get_peers(points, point, pacemaker.pose)
                point.peer_state = p_state

                min_dist_plus, min_dist_minus, has_peer_forward, has_peer_back = get_nearest_distances(peers)

                force = control_force(min_dist_plus, min_dist_minus, has_peer_forward, has_peer_back, point)
                point.apply_force(force)
                point.step(dt)

        pacemaker.step()


def circle_transform(pose, radius=8.0):
    phi = pose / radius

    return np.array([
        radius * np.cos(phi),
        radius * np.sin(phi)
    ])


def animate(i):
    nums = data[i]
    points = np.array(nums).reshape(-1, n_params)
    pacemaker_pose = pacemaker_motion_func(i * dt)
    circle_colors = ["blue", "green", "red"]
    plt.clf()

    plt.xlabel('x')
    plt.xlim((-9, 9))
    plt.ylim((-9, 9))
    plt.ylabel('y')

    plt.gca().set_aspect('equal', adjustable='box')
    traj_circle = Circle(xy=(0, 0), radius=8.0, color="green", fill=False)
    plt.gca().add_artist(traj_circle)

    x_pacemaker, y_pacemaker = circle_transform(pacemaker_pose)
    plt.scatter(x_pacemaker, y_pacemaker, s=s_anchor, color="red")

    for point in points:
        pose, dpose, u, peer_state, id, is_alive = point
        
        if int(is_alive) > 0:
            if colorize_perception_area:
                circle_color = circle_colors[int(peer_state)]
            else:
                circle_color = "grey"

            x, y = circle_transform(pose)
            plt.scatter(x, y, s=s_point, color="blue")
            circle = Circle(xy=(x, y), radius=R_vis, color=circle_color, alpha=0.2)

            plt.gca().add_artist(circle)


def create_gif(log_path, gif_path, frame_step=4):
    print(f"creating gif {gif_path}")
    iters = 0

    with open(log_path, "r") as file:
        lines = [line for line in file]

        for line in lines:
            nums = [float(item) for item in line.rstrip().split(' ')]
            data.append(nums)
            iters += 1

            if iters >= n_frames:
                break
        
        fig, ax = plt.subplots()
        anim = animation.FuncAnimation(fig, animate, frames = tqdm(range(0, n_frames, frame_step)), interval = 20)
        anim.save(gif_path, fps = 60, writer = 'pillow')


def plot_graph(log_path, graph_path):
    print(f"plotting graph: {graph_path}")

    with open(log_path, "r") as file:
        line_num = 0
        t_prev, zs_prev, alives_prev = 0, [], []

        fig = plt.figure(figsize = (20, 20))
        ax = fig.add_subplot()

        for line in tqdm(file):
            t = line_num * dt

            nums = [float(item) for item in line.rstrip().split(' ')]
            points = np.array(nums).reshape(-1, n_params)
            pacemaker_pose = pacemaker_motion_func(t)

            dists, alives = [], []

            for point in points:
                pose, dpose, u, peer_state, id, is_alive = point
                dist = pacemaker_pose - pose
                dists.append(dist)
                alives.append(int(is_alive))

            if line_num > 0:
                for i, (z_diff, z_prev, alive, alive_prev) in enumerate(zip(dists, zs_prev, alives, alives_prev)):
                    if alive > 0 and alive_prev > 0:
                        ax.plot((t_prev, t), (z_prev, z_diff), color=points_colors[i])

            zs_prev = dists
            t_prev = t
            alives_prev = alives

            line_num += 1

        plt.xlabel('t', fontsize=14)
        plt.ylabel(r'$x_i - x_{pacemaker}$', fontsize=14)
        plt.title(r'$x_i - x_{pacemaker}$ for all agents', fontsize=14)
        #plt.ylim(z_bounds)
        
        plt.savefig(graph_path)
        plt.show()


def run(scene_name, n_points):
    log_path = f"logs/cruise_control/log_{scene_name}.txt"
    gif_path = f"gifs/{scene_name}.gif"
    graph_path = f"gifs/{scene_name}.jpg"

    process(n_points, log_path)
    create_gif(log_path, gif_path, frame_step=10)
    plot_graph(log_path, graph_path)

run("cruise_control_acc", 10)