import numpy as np
import time

U_MAX = 2.0


def clamp(value):
    if value < -U_MAX:
        return -U_MAX
    else:
        if value > U_MAX:
            return U_MAX
        else:
            return value

class PID:
    def __init__(self, dt=0.008, kp=1.0, kd=0.0, ki=0.0):
        self.kp = kp
        self.kd = kd
        self.ki = ki

        self.e_prev = 0.0
        self.e_integral = 0.0
        self.dt = dt
        self.u_prev = 0


    def set_coefs(self, kp, kd, ki):
        self.kp = kp
        self.kd = kd
        self.ki = ki


    def get_u(self, e):
        e_diff = (e - self.e_prev) / self.dt
        #print('e = ' + str(e) + ', e_prev = ' + str(self.e_prev) + ', de = ' + str(e_diff))
        u = self.kp * e + self.kd * e_diff + self.ki * self.e_integral

        self.e_integral += e
        self.e_prev = e
        self.u_prev = u

        return clamp(u)
    

    def get_u_by_dt(self, e, dt):
        e_diff = (e - self.e_prev) / dt
        #print('e = ' + str(e) + ', e_prev = ' + str(self.e_prev) + ', de = ' + str(e_diff))
        u = self.kp * e + self.kd * e_diff + self.ki * self.e_integral

        self.e_integral += e
        self.e_prev = e
        self.u_prev = u

        return u
    

class SigmaControl:
    def __init__(self, max_value, dt=0.008):
        self.max_value = max_value
        self.e_prev = 0.0
        self.e_integral = 0.0
        self.dt = dt
        self.u_prev = 0
        self.du_prev = 0
        self.prev_moment = 0
        self.u_out = 0


    def clamp(self, value):
        if value < -self.max_value:
            return -self.max_value
        else:
            if value > self.max_value:
                return self.max_value
            else:
                return value


    def get_u(self, e, d, id):
        k = 1
        u = (np.exp(e*k) - np.exp(-e*k)) / (np.exp(e*k) + np.exp(-e*k))

        '''if abs(u - self.u_prev) > 1:
            u = 0'''
        
        
        thres = 0.07
        du = u - self.u_prev

        if du > thres:
            u = self.u_prev + thres
        if -du > thres:
            u = self.u_prev - thres
        
        self.e_integral += e
        self.e_prev = e
        self.u_prev = u
        self.du_prev = du

        return u
    

class Sliding:
    def __init__(self, k):
        self.k = k


    def sign(self, value):
        if value >= 0:
            return 1
        return -1


    def get_u(self, e):
        return self.k * self.sign(e)