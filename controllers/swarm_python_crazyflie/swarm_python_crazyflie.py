import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import os

from pid_controller import *

from controller import Robot, Supervisor
from controller import Compass
from controller import GPS
from controller import Gyro
from controller import InertialUnit
from controller import Keyboard
from controller import LED
from controller import Motor


#Global constants
r_perception = 10
x_anchor = np.array([3.0, 3.0, 5.0])                    
w = 0.3
anchor_id = 1

k_u = 5

#################################
def u(e):
    return k_u * e


def get_distances(informants):
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
    
    vectors = [informants[id] for id in ids_set]

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


def control_inputs(imu, altitude, distances, id):
    d_x_plus, d_y_plus, d_z_plus, d_x_minus, d_y_minus, d_z_minus = distances
    roll, pitch, yaw = imu

    e_x_plus = d_x_plus
    e_x_minus = d_x_minus
    e_y_plus = d_y_plus
    e_y_minus = d_y_minus
    e_z_plus = d_z_plus - w
    e_z_minus = d_z_minus - w

    vx = u(e_x_plus) - u(e_x_minus)
    vy = u(e_y_plus) - u(e_y_minus)
    altitude_ref = altitude + u(e_z_plus) - u(e_z_minus)

    desired_state = [0, 0, 0, altitude_ref, vx, vy]
    return desired_state


def main(logging=False, log_id=1):
    robot = Supervisor()
    print('robot initiated')
    timestep = int(robot.getBasicTimeStep())
    dt = robot.getBasicTimeStep() / 1000
    robot_id = int(robot.getName().split('_')[1])
    root_node = robot.getRoot()
    children = root_node.getField('children')
    other_robot_nodes = []

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

        i += 1

    imu = robot.getDevice("inertial_unit")
    imu.enable(timestep)
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)

    keyboard = Keyboard()
    keyboard.enable(timestep)

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

    if robot_id == anchor_id:
        target_altitude = 10.0
    else:
        target_altitude = np.random.uniform(8.0, 10.0)

    if logging:
        os.chdir('../../logs')
        folder_name = "log_" + str(log_id)
        
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

        vx_global, vy_global = controller.get_vxy_global(x, y)
        vx = vx_global * np.cos(yaw) + vy_global * np.sin(yaw)
        vy = -vx_global * np.sin(yaw) + vy_global * np.cos(yaw)

        desired_altitude = target_altitude

        forward_desired = 0
        sideways_desired = 0
        yaw_desired = 0
        height_diff_desired = 0

        #Dealing with the swarming algorithm######################################################
        if robot.getTime() > 8.0:
            informants = []
            x_cur = np.array(gps.getValues())

            for node in other_robot_nodes:
                translation = np.array(node.getField('translation').getSFVec3f())
                vec = translation - x_cur

                if robot.getTime() > 25.0:
                    print('r_perception reduced!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    r_perc = 0.9
                else:
                    r_perc = r_perception

                if np.linalg.norm(vec) <= r_perc:
                    informants.append(vec)
            
            if anchor_id == -1:
                anchor_vec = x_anchor - x_cur

                if np.linalg.norm(anchor_vec) <= r_perception:
                    informants.append(anchor_vec)
            
            if robot_id != anchor_id:
                distances, vectors = get_distances(informants)
                desired_state = control_inputs(imu_values, altitude, distances, robot_id)
            else:
                distances = [0 for i in range(6)]
                key = keyboard.getKey()

                while key > 0:
                    if key == Keyboard.UP:
                        forward_desired = 0.5
                    elif key == Keyboard.DOWN:
                        forward_desired = -0.5
                    elif key == Keyboard.RIGHT:
                        sideways_desired = -0.5
                    elif key == Keyboard.LEFT:
                        sideways_desired = +0.5
                    elif key == ord('Q'):
                        yaw_desired = 1.0
                    elif key == ord('E'):
                        yaw_desired = -1.0
                    elif key == ord('W'):
                        height_diff_desired = 0.1
                    elif key == ord('S'):
                        height_diff_desired = -0.1

                    key = keyboard.getKey()

                desired_altitude += height_diff_desired * dt
                desired_state = [0, 0, yaw_desired, desired_altitude, forward_desired, sideways_desired]

            #Logging the robot`s parameters
            if logging:
                x, y, z = gps.getValues()
                #print('distances_to_log: ' + str(distances_to_log))
                d_x_plus, d_y_plus, d_z_plus, d_x_minus, d_y_minus, d_z_minus = distances
                #line = str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(roll) + ' ' + str(pitch) + ' ' + str(yaw) + ' ' + str(roll_disturbance) + ' ' + str(pitch_disturbance) + ' ' + str(yaw_disturbance) + ' ' + str(target_altitude) + ' ' + str(d_x_plus) + ' ' + str(d_x_minus) + ' ' + str(d_y_plus) + ' ' + str(d_y_minus) + ' ' + str(d_z_plus) + ' ' + str(d_z_minus) + '\n'
                #file.write(line)
            #######################################################################################
        else:
            desired_state = [0, 0, 0, target_altitude, 0, 0]
        
        actual_state = [roll, pitch, dyaw, altitude, vx, vy]
        m1, m2, m3, m4 = controller.controller(actual_state, desired_state)

        m1_motor.setVelocity(-m1)
        m2_motor.setVelocity(m2)
        m3_motor.setVelocity(-m3)
        m4_motor.setVelocity(m4)

    if logging:
        file.close()


if __name__ == '__main__':
    main(logging=False, log_id=5)