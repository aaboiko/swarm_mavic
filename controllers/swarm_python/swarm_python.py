import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

from controller import Robot, Supervisor
from controller import Compass
from controller import GPS
from controller import Gyro
from controller import InertialUnit
from controller import Keyboard
from controller import LED
from controller import Motor

from control import PID, Sliding

#Global constants
k_vertical_thrust = 68.5
k_vertical_offset = 0.6
k_vertical_p = 3.0
k_roll_p = 50.0
k_pitch_p = 30.0

r_perception = np.inf
x_anchor = np.array([3.0, 3.0, 5.0])                    
w = 1.0
anchor_id = 1

pitch_disturbance_max = 2.0
pitch_disturbance_min = -2.0

kp_alt = 2
kd_alt = 0.0
ki_alt = 0.0

kp_roll = 1
kd_roll = 0.0
ki_roll = 0.0

kp_pitch = 1
kd_pitch = 0.0
ki_pitch = 0.0

kp_yaw = 1
kd_yaw = 0.0
ki_yaw = 0.0

pid_alt = PID(kp=kp_alt, kd=kd_alt, ki=ki_alt)
pid_roll = PID(kp=kp_roll, kd=kd_roll, ki=ki_roll)
pid_pitch = PID(kp=kp_pitch, kd=kd_pitch, ki=ki_pitch)
pid_yaw = PID(kp=kp_yaw, kd=kd_yaw, ki=ki_yaw)
#################################

def CLAMP(value, low, high):
    if value < low:
        return low
    else:
        if value > high:
            return high
        else:
            return value
        

def sigma_control(e):
    return 2 * (1 / (1 + np.exp(-e))) - 1


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


def control_inputs(imu, altitude, distances, vectors, type="py"):
    d_x_plus, d_y_plus, d_z_plus, d_x_minus, d_y_minus, d_z_minus = distances
    roll, pitch, yaw = imu

    e_x_plus = d_x_plus
    e_x_minus = d_x_minus
    e_y_plus = d_y_plus
    e_y_minus = d_y_minus
    e_z_plus = d_z_plus - w
    e_z_minus = d_z_minus - w

    if type == "py":
        target_yaw = np.arctan2(e_y_plus - e_y_minus, e_x_plus - e_x_minus)
        direction = 1

        if np.cos(target_yaw - yaw) < 0:
            direction = -1

        e_yaw = target_yaw - yaw

        if e_yaw > np.pi:
            e_yaw -= 2 * np.pi
        if e_yaw < -np.pi:
            e_yaw += 2 * np.pi

        e_pitch = np.sqrt((e_x_plus - e_x_minus)**2 + (e_y_plus - e_y_minus)**2)
        roll_disturbance_ref = 0
        yaw_disturbance_ref = pid_yaw.get_u(e_yaw)
        pitch_disturbance_ref = -direction * pid_pitch.get_u(e_pitch)

        print('e_x_plus = ' + str(e_x_plus) + ', e_x_minus = ' + str(e_x_minus) + ', e_y_plus = ' + str(e_y_plus) + ', e_y_minus = ' + str(e_y_minus) + ', target_yaw = ' + str(target_yaw) + ', yaw = ' + str(yaw) + ', e_yaw = ' + str(e_yaw) + ' e_pitch = ' + str(e_pitch) + ', dir : ' + str(direction))

    if type == "rp":
        pass
    
    altitude_ref = altitude + pid_alt.get_u(e_z_plus) - pid_alt.get_u(e_z_minus)

    return roll_disturbance_ref, pitch_disturbance_ref, yaw_disturbance_ref, altitude_ref
        

def get_motor_moments(imu, gyro, altitude, roll_disturbance, pitch_disturbance, yaw_disturbance, target_altitude):
    roll, pitch, yaw = imu
    roll_velocity, pitch_velocity, yaw_velocity = gyro
    
    roll_input = k_roll_p * CLAMP(roll, -1.0, 1.0) + roll_velocity + roll_disturbance
    pitch_input = k_pitch_p * CLAMP(pitch, -1.0, 1.0) + pitch_velocity + pitch_disturbance
    yaw_input = yaw_disturbance
    clamped_difference_altitude = CLAMP(target_altitude - altitude + k_vertical_offset, -1.0, 1.0)
    vertical_input = k_vertical_p * clamped_difference_altitude**3

    front_left_motor_input = k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input
    front_right_motor_input = k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input
    rear_left_motor_input = k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input
    rear_right_motor_input = k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input

    return front_left_motor_input, front_right_motor_input, rear_left_motor_input, rear_right_motor_input


def main():
    robot = Supervisor()
    print('robot initiated')
    timestep = int(robot.getBasicTimeStep())
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

    front_left_led = robot.getDevice("front left led")
    front_right_led = robot.getDevice("front right led")
    imu = robot.getDevice("inertial unit")
    imu.enable(timestep)
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    compass = robot.getDevice("compass")
    compass.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)

    keyboard = Keyboard()
    keyboard.enable(timestep)

    front_left_motor = robot.getDevice("front left propeller")
    front_right_motor = robot.getDevice("front right propeller")
    rear_left_motor = robot.getDevice("rear left propeller")
    rear_right_motor = robot.getDevice("rear right propeller")
    motors = [front_left_motor, front_right_motor, rear_left_motor, rear_right_motor]

    for motor in motors:
        motor.setPosition(np.inf)
        motor.setVelocity(1.0)

    print("Starting the drone...\n")

    target_altitude = np.random.uniform(1.0, 10.0)

    while robot.step(timestep) != -1:
        time = robot.getTime()

        imu_values = imu.getRollPitchYaw()
        altitude = gps.getValues()[2]
        gyro_values = gyro.getValues()

        led_state = int(time) % 2
        front_left_led.set(led_state)
        front_right_led.set(1 - led_state)

        roll_disturbance = 0.0
        pitch_disturbance = 0.0
        yaw_disturbance = 0.0

        #Dealing with the swarming algorithm######################################################
        if robot.getTime() > 5.0:
            informants = []
            x_cur = np.array(gps.getValues())

            for node in other_robot_nodes:
                translation = np.array(node.getField('translation').getSFVec3f())
                vec = translation - x_cur

                if np.linalg.norm(vec) <= r_perception:
                    informants.append(vec)
            
            if anchor_id == -1:
                anchor_vec = x_anchor - x_cur

                if np.linalg.norm(anchor_vec) <= r_perception:
                    informants.append(anchor_vec)
            
            if robot_id != anchor_id:
                distances, vectors = get_distances(informants)
                roll_disturbance, pitch_disturbance, yaw_disturbance, target_altitude = control_inputs(imu_values, altitude, distances, vectors)
                #print('roll: ' + str(roll_disturbance) + ', pitch: ' + str(pitch_disturbance) + ', yaw: ' + str(yaw_disturbance) + ', alt: ' + str(target_altitude))

        #######################################################################################

        key = keyboard.getKey()

        while key > 0:
            if robot_id == anchor_id:
                if key == Keyboard.UP:
                    pitch_disturbance = -2.0
                elif key == Keyboard.DOWN:
                    pitch_disturbance = 2.0
                elif key == Keyboard.RIGHT:
                    yaw_disturbance = -1.3
                elif key == Keyboard.LEFT:
                    yaw_disturbance = 1.3
                elif key == Keyboard.SHIFT + Keyboard.RIGHT:
                    roll_disturbance = -2.0
                elif key == Keyboard.SHIFT + Keyboard.LEFT:
                    roll_disturbance = 2.0
                elif key == Keyboard.SHIFT + Keyboard.UP:
                    target_altitude += 0.05
                    print("target altitude: " + str(target_altitude))
                elif key == Keyboard.SHIFT + Keyboard.DOWN:
                    target_altitude -= 0.05
                    print("target altitude: " + str(target_altitude))

            key = keyboard.getKey()

        front_left_motor_input, front_right_motor_input, rear_left_motor_input, rear_right_motor_input = get_motor_moments(imu_values, gyro_values, altitude, roll_disturbance, pitch_disturbance, yaw_disturbance, target_altitude)

        front_left_motor.setVelocity(front_left_motor_input)
        front_right_motor.setVelocity(-front_right_motor_input)
        rear_left_motor.setVelocity(-rear_left_motor_input)
        rear_right_motor.setVelocity(rear_right_motor_input)


if __name__ == '__main__':
    main()