import numpy as np
import os

from controller import Robot, Supervisor
from controller import GPS
from controller import Gyro
from controller import InertialUnit
from controller import Motor

#from controllers.swarm_python.control import CustomController


k_vertical_thrust = 68.5
k_vertical_offset = 0.6
k_vertical_p = 3.0

k_roll_p = 50.0
k_pitch_p = 30.0
k_yaw_p = 1

k_roll_d = 0
k_pitch_d = 0
k_yaw_d = 0.0

U_ROLL = 2.0
U_PITCH = 2.0
U_YAW = 1.3
FREQ = 2.5


def CLAMP(value, low, high):
    if value < low:
        return low
    else:
        if value > high:
            return high
        else:
            return value
        

def get_motor_moments(imu, gyro, altitude, roll_disturbance, pitch_disturbance, yaw_disturbance, target_altitude):
    roll, pitch, yaw = imu
    roll_velocity, pitch_velocity, yaw_velocity = gyro
    
    roll_input = k_roll_p * CLAMP(roll, -1.0, 1.0) + roll_velocity + roll_disturbance
    pitch_input = k_pitch_p * CLAMP(pitch, -1.0, 1.0) + pitch_velocity + pitch_disturbance
    #roll_input = k_roll_p * roll + k_roll_d * roll_velocity + roll_disturbance
    #pitch_input = k_pitch_p * pitch + k_pitch_d * pitch_velocity + pitch_disturbance
    yaw_input = yaw_disturbance + k_yaw_d * yaw_velocity
    clamped_difference_altitude = CLAMP(target_altitude - altitude + k_vertical_offset, -1.0, 1.0)
    vertical_input = k_vertical_p * clamped_difference_altitude**3

    front_left_motor_input = k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input
    front_right_motor_input = k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input
    rear_left_motor_input = k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input
    rear_right_motor_input = k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input

    return front_left_motor_input, front_right_motor_input, rear_left_motor_input, rear_right_motor_input


def chatter(time, roll=False, pitch=False, yaw=False):
    A_ROLL = roll * U_ROLL
    A_PITCH = pitch * U_PITCH
    A_YAW = yaw * U_YAW

    roll_disturbance = A_ROLL * np.sin(2 * np.pi * FREQ * time)
    pitch_disturbance = A_PITCH * np.sin(2 * np.pi * FREQ * time)
    yaw_disturbance = A_YAW * np.sin(2 * np.pi * FREQ * time)

    return roll_disturbance, pitch_disturbance, yaw_disturbance


def u(e):
    return np.tanh(e)


def main(logging=False, log_id=1):
    robot = Supervisor()
    print('robot initiated')
    timestep = int(robot.getBasicTimeStep())

    imu = robot.getDevice("inertial unit")
    imu.enable(timestep)
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)

    front_left_motor = robot.getDevice("front left propeller")
    front_right_motor = robot.getDevice("front right propeller")
    rear_left_motor = robot.getDevice("rear left propeller")
    rear_right_motor = robot.getDevice("rear right propeller")
    motors = [front_left_motor, front_right_motor, rear_left_motor, rear_right_motor]

    for motor in motors:
        motor.setPosition(np.inf)
        motor.setVelocity(1.0)

    print("Starting the drone...\n")

    roll_disturbance = 0.0
    pitch_disturbance = 0.0
    yaw_disturbance = 0.0
    target_altitude = 5.0

    if logging:
        file = open("../../logs/test/log_" + str(log_id) + ".txt", "w")

    while robot.step(timestep) != -1:
        time = robot.getTime()

        imu_values = imu.getRollPitchYaw()
        altitude = gps.getValues()[2]
        gyro_values = gyro.getValues()

        if robot.getTime() > 5.0:
            roll_disturbance, pitch_disturbance, yaw_disturbance = chatter(time, roll=False, pitch=True, yaw=False)
            pitch_disturbance = u(pitch_disturbance)

            if logging:
                roll, pitch, yaw = imu_values
                uroll, upitch, uyaw = gyro_values
                line = str(roll) + ' ' + str(pitch) + ' ' + str(yaw) + ' ' + str(uroll) + ' ' + str(upitch) + ' ' + str(uyaw) + ' ' + str(roll_disturbance) + ' ' + str(pitch_disturbance) + ' ' + str(yaw_disturbance) + '\n'
                file.write(line)

        front_left_motor_input, front_right_motor_input, rear_left_motor_input, rear_right_motor_input = get_motor_moments(imu_values, gyro_values, altitude, roll_disturbance, pitch_disturbance, yaw_disturbance, target_altitude)

        front_left_motor.setVelocity(front_left_motor_input)
        front_right_motor.setVelocity(-front_right_motor_input)
        rear_left_motor.setVelocity(-rear_left_motor_input)
        rear_right_motor.setVelocity(rear_right_motor_input)

    if logging:
        file.close()


main(logging=False, log_id=2)