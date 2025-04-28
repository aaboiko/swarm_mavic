import numpy as np
import cv2
import math
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


VELOCITY = 0.03


def main(traj="linear", logging=False, log_name="log"):
    robot = Supervisor()
    print('robot initiated')
    timestep = int(robot.getBasicTimeStep())
    dt = robot.getBasicTimeStep() / 1000

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

    if logging:
        os.chdir('../../logs')
        folder_name = log_name
        
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        
        os.chdir(folder_name)
        file = open("anchor_" + log_name + ".txt", "w")
        print('logging started. Folder name: ' + folder_name)

    target_altitude = 1.0
    
    forward_desired = 0
    sideways_desired = 0
    yaw_desired = 0

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

        if time >= 8.0:
            if traj == "circle":
                forward_desired = VELOCITY * np.cos(2 * np.pi * time / 60)
                sideways_desired = VELOCITY * np.sin(2 * np.pi * time / 60)
            else:
                forward_desired = VELOCITY

        desired_state = [0, 0, yaw_desired, target_altitude, forward_desired, sideways_desired]
        actual_state = [roll, pitch, dyaw, altitude, vx, vy]
        m1, m2, m3, m4 = controller.controller(actual_state, desired_state)

        m1_motor.setVelocity(-m1)
        m2_motor.setVelocity(m2)
        m3_motor.setVelocity(-m3)
        m4_motor.setVelocity(m4)

        if logging:
            d_to_anchor = 0
            line = f"{x} {y} {altitude} {vx_global} {vy_global} {vz_global} {roll} {pitch} {yaw} {droll} {dpitch} {dyaw} {0} {0} {0} {0} {0} {0} {d_to_anchor}\n"
            file.write(line)

    if logging:
        file.close()


main(traj="circle", log_name="")