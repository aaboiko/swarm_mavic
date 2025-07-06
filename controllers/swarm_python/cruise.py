import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from point_mass import Agent_1D, Pacemaker
from matplotlib.patches import Circle
from tqdm import tqdm
from functools import partial

n_params = 6

R_vis = 3.0
w = 0.5
u_max = 10.0
dt = 0.02
traj_len = 10000

n_frames = 10000
data = []

s_point = 30
s_anchor = 60


def sign(x):
    if x >= 0:
        return 1
    
    return -1


def xi(g):
    return np.tanh(g)


def get_peers(points, focus_point, anchor_pose):
    peers = []
    peer_state = 0

    if abs(anchor_pose - focus_point.pose) <= R_vis:
        peer_state = 2
        vec = anchor_pose - focus_point.pose
        peers.append(vec)

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
    
    return peers, peer_state


def get_nearest_distances(peers, anchor_dir):
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

    ids = [id_plus, id_minus]
    mins = [min_dist_plus, min_dist_minus]
    distances = []

    for i in range(2):
        if ids[i] == -1:
            if anchor_dir[i] < 0:
                distances.append(w)
            else:
                distances.append(R_vis)
        else:
            distances.append(mins[i])

    return distances


def control_force(distances, point):
    d_plus, d_minus = distances
    dx = point.dpose
    sigma = dx - xi(d_plus) + xi(d_minus)
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
        interval = np.random.uniform(R_min, R_vis)
        x = cur_x - interval
        point = Agent_1D(id=i, x=x)
        points.append(point)
        cur_x = x

    pacemaker = Pacemaker()

    n_iters = 10000
    open(log_path, "w").close()

    for iter in tqdm(range(n_iters)):
        write_log(log_path, points)

        for point in points:
            if point.is_alive:
                peers, p_state = get_peers(points, point, pacemaker.pose)
                point.peer_state = p_state

                vec_to_pacemaker = pacemaker.pose - point.pose
                point.prior_knowledge = np.sign(vec_to_pacemaker)
                signs = [sign(vec_to_pacemaker), -sign(vec_to_pacemaker)]

                if len(peers) > 0:
                    distances = get_nearest_distances(peers, signs)
                else:
                    distances = []

                    for i, item in enumerate(signs):
                        if item >= 0:
                            distances.append(R_vis * item)
                        else:
                            distances.append(-w * item)

                force = control_force(distances, point)
                point.apply_force(force)
                point.step(dt)

        pacemaker.step()


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
        anim = animation.FuncAnimation(fig, partial(animate_xy, traj_path=traj_path), frames = tqdm(range(0, n_frames, frame_step)), interval = 20)
        anim.save(gif_path, fps = 60, writer = 'pillow')