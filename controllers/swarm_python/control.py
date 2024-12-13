import numpy as np

class PID:
    def __init__(self, dt=0.008, kp=1.0, kd=0.0, ki=0.0):
        self.kp = kp
        self.kd = kd
        self.ki = ki

        self.e_prev = 0.0
        self.e_integral = 0.0
        self.dt = dt


    def get_u(self, e, dt):
        e_diff = (e - self.e_prev) / dt
        u = self.kp * e + self.kd * e_diff + self.ki * self.e_integral

        self.e_integral += e
        self.e_prev = e

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