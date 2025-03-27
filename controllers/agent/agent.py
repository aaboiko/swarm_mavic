import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append("../swarm_python_crazyflie")
from pid_controller import *

from controller import Robot, Supervisor
from controller import Compass
from controller import GPS
from controller import Gyro
from controller import InertialUnit
from controller import Keyboard
from controller import LED
from controller import Motor


R_vis = 0.7
w = 0.5
u_max = 0.5


def sign(x):
    if x >= 0:
        return 1
    
    return -1


def xi(g):
    return np.tanh(g)


def control_force(distances, dpose):
    d_x_plus, d_y_plus, d_z_plus, d_x_minus, d_y_minus, d_z_minus = distances
    vx, vy, vz = dpose

    sigma_x = vx - xi(d_x_plus) + xi(d_x_minus)
    sigma_y = vy - xi(d_y_plus) + xi(d_y_minus)
    sigma_z = vz - xi(d_z_plus) + xi(d_z_minus)

    ux = -u_max * sign(sigma_x)
    uy = -u_max * sign(sigma_y)
    uz = -u_max * sign(sigma_z)

    return np.array([ux, uy, uz])


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


def main(logging=False, log_name="log"):
    robot = Supervisor()
    print('robot initiated')
    timestep = int(robot.getBasicTimeStep())
    dt = robot.getBasicTimeStep() / 1000

    robot_id = int(robot.getName().split('_')[1])
    root_node = robot.getRoot()
    children = root_node.getField('children')
    other_robot_nodes = []
    anchor_node = None

    i = 0
    while True:
        node = children.getMFNode(i)

        if node is None:
            break

        name_field = node.getField('name')

        if name_field is not None:
            name = name_field.getSFString()

            if name.split('_')[0] == 'robot' and int(name.split('_')[1]) != robot_id:
                other_robot_nodes.append(node)
            if name.split('_')[0] == 'robot' and int(name.split('_')[1]) == 0:
                anchor_node = node

        i += 1

    imu = robot.getDevice("inertial_unit")
    imu.enable(timestep)
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)

    m1_motor = robot.getDevice("m1_motor")
    m1_motor.setPosition(np.inf)
    m1_motor.setVelocity(-1.0)

    m2_motor = robot.getDevice("m2_motor")
    m2_motor.setPosition(np.inf)
    m2_motor.setVelocity(1.0)

    m3_motor = robot.getDevice("m3_motor")
    m3_motor.setPosition(np.inf)
    m3_motor.setVelocity(-1.0)

    m4_motor = robot.getDevice("m4_motor")
    m4_motor.setPosition(np.inf)
    m4_motor.setVelocity(1.0)

    controller = PID(dt=dt)

    print("Starting the drone...\n")

    target_altitude = np.random.uniform(3.0, 7.0)

    target_altitude_speed = 0.0
    forward_desired = 0
    sideways_desired = 0

    if logging:
        os.chdir('../../logs')
        folder_name = log_name
        
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        
        os.chdir(folder_name)
        file = open("robot_" + str(robot_id) + ".txt", "w")
        print('logging started. Folder name: ' + folder_name)

    while robot.step(timestep) != -1:
        time = robot.getTime()

        imu_values = imu.getRollPitchYaw()
        roll, pitch, yaw = imu_values
        x, y, altitude = gps.getValues()
        droll, dpitch, dyaw = gyro.getValues()

        dpose_global = controller.get_vxyz_global(x, y, altitude)
        vx_global, vy_global, vz_global = dpose_global
        vx = vx_global * np.cos(yaw) + vy_global * np.sin(yaw)
        vy = -vx_global * np.sin(yaw) + vy_global * np.cos(yaw)

        target_altitude_acc = 0.0
        forward_acc_desired = 0.0
        sideways_acc_desired = 0.0
        yaw_desired = 0

        if time > 8.0:
            peers = []
            x_cur = np.array(gps.getValues())

            for node in other_robot_nodes:
                translation = np.array(node.getField('translation').getSFVec3f())
                vec = translation - x_cur

                if np.linalg.norm(vec) <= R_vis:
                    peers.append(vec)

            anchor_pose = np.array(anchor_node.getField('translation').getSFVec3f())
            vec_to_anchor = anchor_pose - np.array([x, y, altitude])
            x_to_anchor, y_to_anchor, z_to_anchor = vec_to_anchor
            signs = [sign(x_to_anchor), sign(y_to_anchor), sign(z_to_anchor), -sign(x_to_anchor), -sign(y_to_anchor), -sign(z_to_anchor)]

            distances = get_nearest_distances(peers, signs)
            
            forward_acc_desired, sideways_acc_desired, target_altitude_acc = control_force(distances, dpose_global)
            target_altitude_speed += target_altitude_acc * dt
            target_altitude += target_altitude_speed * dt + 0.5 * target_altitude_acc * dt**2

            forward_desired += forward_acc_desired * dt
            sideways_desired += sideways_acc_desired * dt

            desired_state = [0, 0, yaw_desired, target_altitude, forward_desired, sideways_desired]

            if logging:
                d_to_anchor = np.linalg.norm(vec_to_anchor)
                line = f"{x} {y} {altitude} {vx_global} {vy_global} {vz_global} {roll} {pitch} {yaw} {droll} {dpitch} {dyaw} {forward_acc_desired} {sideways_acc_desired} {target_altitude_acc} {vec_to_anchor[0]} {vec_to_anchor[1]} {vec_to_anchor[2]} {d_to_anchor}\n"
                file.write(line)

        else:
            desired_state = [0, 0, yaw_desired, target_altitude, forward_desired, sideways_desired]

        actual_state = [roll, pitch, dyaw, altitude, vx, vy]
        m1, m2, m3, m4 = controller.controller(actual_state, desired_state)

        m1_motor.setVelocity(-m1)
        m2_motor.setVelocity(m2)
        m3_motor.setVelocity(-m3)
        m4_motor.setVelocity(m4)

    if logging:
        file.close()


if __name__ == '__main__':
    main(logging=False, log_name="log")