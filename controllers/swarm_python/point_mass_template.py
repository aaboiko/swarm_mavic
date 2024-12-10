import numpy as np
import matplotlib.pyplot as plt
import time
#import keyboard

from control import PID, Sliding
from point_mass import PointMass, PointsHandler

N_POINTS = 10

x_bound_min = -10
x_bound_max = 10
y_bound_min = -10
y_bound_max = 10
z_bound_min = 0
z_bound_max = 20

r_perception = 30.0
x_anchor = np.array([3.0, 3.0, 1.0])                    
w = 1.0
anchor_id = -1

kp = 0.1
kd = 0.1
ki = 0.1

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
    
    #distances = [informants[id] for id in ids_set]
    distances = [informants[id] for id in ids]
    
    return distances


def control_force(dt, distances):
    d_x_plus, d_y_plus, d_z_plus, d_x_minus, d_y_minus, d_z_minus = distances

    e_x_plus = d_x_plus
    e_x_minus = d_x_minus
    e_y_plus = d_y_plus
    e_y_minus = d_y_minus
    e_z_plus = d_z_plus - w
    e_z_minus = d_z_minus - w

    pid = PID(dt=dt, kp=kp, kd=kd, ki=ki)

    ux = pid.get_u(e_x_plus) - pid.get_u(e_x_minus)
    uy = pid.get_u(e_y_plus) - pid.get_u(e_y_minus)
    uz = pid.get_u(e_z_plus) - pid.get_u(e_z_minus)

    return np.array([ux, uy, uz])


def get_informant(focus_point):
    print('\nPOints: ')
    for p in points:
        print('pose: ' + str(p.pose))
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


def draw(points):
    plt.xlim((x_bound_min, x_bound_max))
    plt.ylim((y_bound_min, y_bound_max))

    x_anc, y_anc, z_anc = x_anchor
    plt.scatter(x_anc, y_anc, s=2, color="red")

    for point in points:
        x, y, z = point.pose
        plt.scatter(x, y, s=1, color="blue")


def process(dt):
    for point in points:
        informant = get_informant(point)
        distances = get_nearest_distances(informant)
        #print(len(distances))
        #print(distances)
        force = control_force(dt, distances)

        point.apply_force(force)
        point.step(dt)

        plt.clf()
        draw(points)
        plt.pause(0.01)


def main():
    for i in range(N_POINTS):
        x = np.random.uniform(x_bound_min, x_bound_max)
        y = np.random.uniform(y_bound_min, y_bound_max)
        z = np.random.uniform(z_bound_min, z_bound_max)
        point = PointMass(id=i, x=x, y=y, z=z)
        points.append(point)
    
    moment_prev = time.time()

    while(True):
        moment = time.time()
        dt = moment - moment_prev

        if dt < 0.001:
            continue

        process(dt)
        moment_prev = moment

main()
plt.show()