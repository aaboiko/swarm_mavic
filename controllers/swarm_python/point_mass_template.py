import numpy as np
import matplotlib.pyplot as plt
import time
import keyboard

from control import PID, Sliding
from point_mass import PointMass

N_POINTS = 1

x_bound_min = -10
x_bound_max = 10
y_bound_min = -10
y_bound_max = 10
z_bound_min = -1
z_bound_max = 10

r_perception = np.inf
x_anchor = np.array([0.0, 0.0, 0.0])                    
w = 1.0
anchor_id = -1

kp = 1
kd = 1
ki = 10

pid_x = PID(kp=kp, kd=kd, ki=ki)
pid_y = PID(kp=kp, kd=kd, ki=ki)
pid_z = PID(kp=kp, kd=kd, ki=ki)

s_point = 10
s_anchor = 20

points = []

def get_nearest_distances(informants):
    min_dist_x_plus, min_dist_y_plus, min_dist_z_plus = np.inf, np.inf, np.inf
    min_dist_x_minus, min_dist_y_minus, min_dist_z_minus = np.inf, np.inf, np.inf
    id_x_plus, id_y_plus, id_z_plus = -1, -1, -1
    id_x_minus, id_y_minus, id_z_minus = -1, -1, -1

    for i in range(len(informants)):
        informant = informants[i]
        vx, vy, vz = informant
        #print('informant = ' + str(informant))

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
    
    if len(informants) > 0:
        vectors = [informants[id] for id in ids]
        #vectors = [informants[id] for id in ids_set]
    else:
        vectors = []

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
    
    return distances, vectors


def control_force(dt, distances, vectors):
    d_x_plus, d_y_plus, d_z_plus, d_x_minus, d_y_minus, d_z_minus = distances
    vec_x_plus, vec_y_plus, vec_z_plus, vec_x_minus, vec_y_minus, vec_z_minus = vectors

    e_z_plus = d_z_plus - w
    e_z_minus = d_z_minus - w

    ux = pid_x.get_u(d_x_plus, dt) - pid_x.get_u(d_x_minus, dt)
    uy = pid_y.get_u(d_y_plus, dt) - pid_y.get_u(d_y_minus, dt)
    uz = pid_z.get_u(e_z_plus, dt) - pid_z.get_u(e_z_minus, dt)

    return np.array([ux, uy, uz])


def get_informant(focus_point):
    informant = []

    if np.linalg.norm(x_anchor - focus_point.pose) <= r_perception and anchor_id == -1:
        vec = x_anchor - focus_point.pose
        informant.append(vec)

    for point in points:
        d = np.linalg.norm(point.pose - focus_point.pose)

        if d <= r_perception and point.id != focus_point.id:
            vec = point.pose - focus_point.pose
            #print('vec: ' + str(vec) + ', point_pose: ' + str(point.pose) + ', focus_pose: ' + str(focus_point.pose))
            informant.append(vec)
    
    return informant


def draw(points, axes='xy'):
    if axes == 'xy':
        plt.xlim((x_bound_min, x_bound_max))
        plt.ylim((y_bound_min, y_bound_max))
    if axes == 'xz':
        plt.xlim((x_bound_min, x_bound_max))
        plt.ylim((z_bound_min, z_bound_max))
    if axes == 'yz':
        plt.xlim((y_bound_min, y_bound_max))
        plt.ylim((z_bound_min, z_bound_max))

    x_anc, y_anc, z_anc = x_anchor

    if axes == 'xy':
        plt.scatter(x_anc, y_anc, s=s_anchor, color="red")

        for point in points:
            x, y, z = point.pose
            plt.scatter(x, y, s=s_point, color="blue")

    if axes == 'xz':
        plt.scatter(x_anc, z_anc, s=s_anchor, color="red")

        for point in points:
            x, y, z = point.pose
            plt.scatter(x, z, s=s_point, color="blue")

    if axes == 'yz':
        plt.scatter(y_anc, z_anc, s=s_anchor, color="red")

        for point in points:
            x, y, z = point.pose
            plt.scatter(y, z, s=s_point, color="blue")


def process(dt, file):
    for point in points:
        informant = get_informant(point)
        distances, vectors = get_nearest_distances(informant)
        #print(len(distances))
        #print(distances)
        force = control_force(dt, distances, vectors)

        #point.apply_force(force)
        point.set_dpose(force)
        point.step(dt)

        plt.clf()
        draw(points, axes='xz')
        plt.pause(0.01)

        x, y, z = point.pose
        file.write(str(x) + ' ' + str(y) + ' ' + str(z) + ' ')

    file.write('\n')


def main():
    for i in range(N_POINTS):
        x = np.random.uniform(x_bound_min, x_bound_max)
        y = np.random.uniform(y_bound_min, y_bound_max)
        z = np.random.uniform(z_bound_min, z_bound_max)
        point = PointMass(id=i, x=x, y=y, z=z)
        points.append(point)
    
    moment_prev = time.time()
    running = True
    
    with open("logs/log_1.txt", "w") as file:
        while(running):
            if keyboard.is_pressed('p'):
                #running = False
                print("p pressed")

            moment = time.time()
            dt = moment - moment_prev

            if dt < 0.001:
                continue

            process(dt, file)
            moment_prev = moment


main()
plt.show()