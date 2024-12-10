import numpy as np

class PointMass:
    def __init__(self, id=0, mass=1.0, x=0, y=0, z=0):
        self.id = id
        self.mass = mass
        self.pose = np.array([x, y, z])
        self.dpose = np.zeros(3)
        self.ddpose = np.zeros(3)
        self.force = np.zeros(3)


    def apply_force(self, force):
        self.force = force


    def step(self, dt):
        self.ddpose = self.force / self.mass
        dpose = self.dpose + self.ddpose * dt
        self.dpose = dpose
        
        pose = self.pose + self.dpose * dt
        self.pose = pose


class PointsHandler:
    def __init__(self, n):
        points = []

        for i in range(n):
            x = np.random.uniform(-10, 10)
            y = np.random.uniform(-10, 10)
            z = np.random.uniform(0, 20)
            point = PointMass(id=i, x=x, y=y, z=z)
            points.append(point)

        self.points = points