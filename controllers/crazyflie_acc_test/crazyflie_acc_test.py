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

u_max = 0.5


def main():
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

    target_altitude = 3.0
    target_altitude_speed = 0.0

    forward_desired = 0
    sideways_desired = 0

    while robot.step(timestep) != -1:
        time = robot.getTime()

        imu_values = imu.getRollPitchYaw()
        roll, pitch, yaw = imu_values
        x, y, altitude = gps.getValues()
        droll, dpitch, dyaw = gyro.getValues()

        vx_global, vy_global = controller.get_vxy_global(x, y)
        vx = vx_global * np.cos(yaw) + vy_global * np.sin(yaw)
        vy = -vx_global * np.sin(yaw) + vy_global * np.cos(yaw)

        target_altitude_acc = 0.0
        forward_acc_desired = 0.0
        sideways_acc_desired = 0.0
        yaw_desired = 0

        key = keyboard.getKey()

        while key > 0:
            if key == Keyboard.UP:
                forward_acc_desired = +u_max
            elif key == Keyboard.DOWN:
                forward_acc_desired = -u_max
            elif key == Keyboard.RIGHT:
                sideways_acc_desired = -u_max
            elif key == Keyboard.LEFT:
                sideways_acc_desired = +u_max
            elif key == ord('W'):
                target_altitude_acc = +u_max
            elif key == ord('S'):
                target_altitude_acc = -u_max

            key = keyboard.getKey()

        target_altitude_speed += target_altitude_acc * dt
        target_altitude += target_altitude_speed * dt + 0.5 * target_altitude_acc * dt**2

        forward_desired += forward_acc_desired * dt
        sideways_desired += sideways_acc_desired * dt

        desired_state = [0, 0, yaw_desired, target_altitude, forward_desired, sideways_desired]

        actual_state = [roll, pitch, dyaw, altitude, vx, vy]
        m1, m2, m3, m4 = controller.controller(actual_state, desired_state)

        m1_motor.setVelocity(-m1)
        m2_motor.setVelocity(m2)
        m3_motor.setVelocity(-m3)
        m4_motor.setVelocity(m4)


if __name__ == '__main__':
    main()