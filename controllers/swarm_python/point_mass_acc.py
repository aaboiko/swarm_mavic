import numpy as np
import matplotlib.pyplot as plt
import time
import os

from point_mass import PointMass
from matplotlib.patches import Circle

N_POINTS = 4
log_path = "logs/point_mass/log_acc_2.txt"

R_vis = 2.0
x_anchor = np.array([0.0, 0.0, 1.0]) 
w = 1.0
u_max = 0.1
dt = 0.02

s_point = 10
s_anchor = 20


def sign(x):
    if x >= 0:
        return 1
    
    return -1


def xi(g):
    return np.tanh(g)


def get_peers(points, focus_point):
    peers = []

    if np.linalg.norm(x_anchor - focus_point.pose) <= R_vis:
        vec = x_anchor - focus_point.pose
        peers.append(vec)

    for point in points:
        d = np.linalg.norm(point.pose - focus_point.pose)

        if d <= R_vis and point.id != focus_point.id:
            vec = point.pose - focus_point.pose
            peers.append(vec)
    
    return peers


def get_nearest_distances(peers):
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
            if i == 2 or i == 5:
                distances.append(w)
            else:
                distances.append(0)
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
            has_peer = int(point.has_peer)
            line += f"{x} {y} {z} {dx} {dy} {dz} {ux} {uy} {uz} {has_peer} "

        file.write(line + "\n")
        print(line)


def process():
    points = [
        PointMass(id=0, x=-2, y=-2, z=int(np.random.uniform(2, 10))),
        PointMass(id=1, x=-2, y=2, z=int(np.random.uniform(2, 10))),
        PointMass(id=2, x=2, y=-2, z=int(np.random.uniform(2, 10))),
        PointMass(id=3, x=2, y=2, z=int(np.random.uniform(2, 10)))
    ]

    n_iters = 10000

    while n_iters > 0:
        n_iters -= 1
        write_log(log_path, points)

        for point in points:
            peers = get_peers(points, point)

            if len(peers) > 0:
                distances = get_nearest_distances(peers)
                point.has_peer = True
            else:
                point.has_peer = False
                vec_to_anchor = x_anchor - point.pose
                x_to_anchor, y_to_anchor, z_to_anchor = vec_to_anchor
                point.prior_knowledge = np.array([sign(x_to_anchor), sign(y_to_anchor), sign(z_to_anchor)])
                signs = [sign(x_to_anchor), sign(y_to_anchor), sign(z_to_anchor), -sign(x_to_anchor), -sign(y_to_anchor), -sign(z_to_anchor)]
                distances = []

                for i, item in enumerate(signs):
                    if item >= 0:
                        distances.append(R_vis * item)
                    else:
                        if i == 2 or i == 5:
                            distances.append(-w * item)
                        else:
                            distances.append(0)

                #print(f"x = {x_to_anchor}, y = {y_to_anchor}, z = {z_to_anchor}")

            #print(f"distances: {distances}")
            #print(f"peers: {peers}")

            force = control_force(distances, point)
            #print(f"force: {force}")

            point.apply_force(force)
            point.step(dt)


def animate_from_log(log_path, axes="xy"):
    with open(log_path, "r") as file:
        for line in file:
            nums = [float(item) for item in line.rstrip().split(' ')]
            points = np.array(nums).reshape(-1, 10)
            x_anc, y_anc, z_anc = x_anchor
            circle_colors = ["blue", "green"]
            plt.clf()

            if axes == "xy":
                plt.xlim((-10, 10))
                plt.ylim((-10, 10))
            if axes == "xz":
                plt.xlim((-5, 5))
                plt.ylim((0, 10))

            plt.gca().set_aspect('equal', adjustable='box')

            if axes == "xy":
                plt.scatter(x_anc, y_anc, s=s_anchor, color="red")
            if axes == "xz":
                plt.scatter(x_anc, z_anc, s=s_anchor, color="red")

            for point in points:
                x, y, z, dx, dy, dz, ux, uy, uz, has_peer = point
                circle_color = circle_colors[int(has_peer)]

                if axes == "xy":
                    plt.scatter(x, y, s=s_point, color="blue")
                    circle = Circle(xy=(x, y), radius=R_vis, color=circle_color, alpha=0.2)
                if axes == "xz":
                    plt.scatter(x, z, s=s_point, color="blue")
                    circle = Circle(xy=(x, z), radius=R_vis, color=circle_color, alpha=0.2)

                plt.gca().add_artist(circle)

            plt.pause(0.01)

        plt.show()


#process()
animate_from_log(log_path, axes="xz")