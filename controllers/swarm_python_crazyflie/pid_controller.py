import numpy as np

kp_att_y = 1
kd_att_y = 0.5

kp_att_rp = 0.5
kd_att_rp = 0.1

kp_vel_xy = 2
kd_vel_xy = 0.5

kp_z = 4
ki_z = 0.1
kd_z = 0.0

kp_yaw = 1.0


def constrain(value, minVal, maxVal):
    return min(maxVal, max(minVal, value))


class PID:
    def __init__(self, dt = 0.008):
        self.dt = dt

        self.roll_desired = 0
        self.pitch_desired = 0
        self.dyaw_desired = 0
        self.altitude_desired = 1.0
        self.vx_desired = 0
        self.vy_desired = 0

        self.roll_command = 0
        self.pitch_command = 0
        self.yaw_command = 0
        self.altitude_command = 0

        self.pastAltitudeError = 0
        self.pastYawRateError = 0
        self.pastPitchError = 0
        self.pastRollError = 0
        self.pastYawError = 0
        self.pastVxError = 0
        self.pastVyError = 0
        self.altitudeIntegrator = 0

        self.pastX = 0
        self.pastY = 0


    def get_vxy_global(self, x, y):
        vx = (x - self.pastX) / self.dt
        vy = (y - self.pastY) / self.dt

        self.pastX = x
        self.pastY = y

        return vx, vy


    def controller(self, actual_state, desired_state):
        roll, pitch, dyaw, altitude, vx, vy = actual_state

        roll_desired, pitch_desired, dyaw_desired, altitude_desired, vx_desired, vy_desired = desired_state
        self.roll_desired = roll_desired
        self.pitch_desired = pitch_desired
        self.dyaw_desired = dyaw_desired
        self.altitude_desired = altitude_desired
        self.vx_desired = vx_desired
        self.vy_desired = vy_desired
        #print('actual state: roll = ' + str(roll) + ', pitch = ' + str(pitch) + ', yaw = ' + str(yaw) + ', dyaw = ' + str(dyaw) + ', alt = ' + str(altitude) + ', vx = ' + str(vx) + ', vy = ' + str(vy))
        #print('desired state: alt_ref = ' + str(altitude_ref) + ', vx = ' + str(vx_ref) + ', vy = ' + str(vy_ref) + ', dyaw = ' + str(yaw_ref))

        self.horizontal_velocity_controller(vx, vy)
        self.fixed_height_controller(altitude)
        self.attitude_controller(roll, pitch, dyaw)
        m1, m2, m3, m4 = self.motor_mixing()

        return m1, m2, m3, m4
    

    def get_u(self, actual_state, desired_state):
        roll, pitch, dyaw, altitude, vx, vy = actual_state

        roll_desired, pitch_desired, dyaw_desired, altitude_desired, vx_desired, vy_desired = desired_state
        self.roll_desired = roll_desired
        self.pitch_desired = pitch_desired
        self.dyaw_desired = dyaw_desired
        self.altitude_desired = altitude_desired
        self.vx_desired = vx_desired
        self.vy_desired = vy_desired




    def horizontal_velocity_controller(self, vx, vy):
        vxError = self.vx_desired - vx
        vxDerivative = (vxError - self.pastVxError) / self.dt
        vyError = self.vy_desired - vy
        vyDerivative = (vyError - self.pastVyError) / self.dt
        #yawError = yaw_ref - yaw

        pitchCommand = kp_vel_xy * constrain(vxError, -1, 1) + kd_vel_xy * vxDerivative
        rollCommand = -kp_vel_xy * constrain(vyError, -1, 1) - kd_vel_xy * vyDerivative

        self.pastVxError = vxError
        self.pastVyError = vyError

        #print('roll_cmd = ' + str(rollCommand) + ', pitch_cmd = ' + str(pitchCommand) + ', yaw_cmd = ')

        self.roll_desired = rollCommand
        self.pitch_desired = pitchCommand


    def fixed_height_controller(self, altitude):
        altitudeError = self.altitude_desired - altitude
        altitudeDerivativeError = (altitudeError - self.pastAltitudeError) / self.dt
        #altitude_command = kp_z * constrain(altitudeError, -1, 1) + kd_z * altitudeDerivativeError + ki_z

        self.altitudeIntegrator += altitudeError * self.dt
        self.altitude_command = kp_z * constrain(altitudeError, -1, 1) + kd_z * altitudeDerivativeError + ki_z * self.altitudeIntegrator + 48
        self.pastAltitudeError = altitudeError

        #print('alt_cmd = ' + str(altitude_command) + ', alt_integrator = ' + str(self.altitudeIntegrator))
        
    

    def attitude_controller(self, roll, pitch, dyaw):
        pitchError = self.pitch_desired - pitch
        pitchDerivativeError = (pitchError - self.pastPitchError) / self.dt
        rollError = self.roll_desired - roll
        rollDerivativeError = (rollError - self.pastRollError) / self.dt
        yawRateError = self.dyaw_desired - dyaw

        self.roll_command = kp_att_rp * constrain(rollError, -1, 1) + kd_att_rp * rollDerivativeError
        self.pitch_command = -kp_att_rp * constrain(pitchError, -1, 1) - kd_att_rp * pitchDerivativeError
        self.yaw_command = kp_att_y * constrain(yawRateError, -1, 1)

        self.pastPitchError = pitchError
        self.pastRollError = rollError
        self.pastYawRateError = yawRateError

        #print('roll = ' + str(roll_out) + ', pitch = ' + str(pitch_out) + ', yaw = ' + str(yaw_out) + ', e_roll = ' + str(rollError) + ', e_pitch = ' + str(pitchError) + ', e_yaw = ' + str(yawRateError) + ', roll_desired = ' + str(roll_desired))
    

    def motor_mixing(self):
        m1 = self.altitude_command - self.roll_command + self.pitch_command + self.yaw_command
        m2 = self.altitude_command - self.roll_command - self.pitch_command - self.yaw_command
        m3 = self.altitude_command + self.roll_command - self.pitch_command + self.yaw_command
        m4 = self.altitude_command + self.roll_command + self.pitch_command - self.yaw_command

        return m1, m2, m3, m4


    

    
