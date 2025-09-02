import numpy as np

class PointMass:
    def __init__(self, id=0, mass=1.0, x=0, y=0, z=0, is_alive=True):
        self.id = id
        self.mass = mass
        self.pose = np.array([x, y, z])
        self.dpose = np.zeros(3)
        self.ddpose = np.zeros(3)
        self.force = np.zeros(3)

        self.has_peer = False
        self.peer_state = 0
        self.prior_knowledge = np.zeros(3)
        self.is_alive = is_alive


    def apply_force(self, force):
        self.force = force


    def set_dpose(self, dpose):
        self.dpose = dpose


    def kill(self):
        self.is_alive = False
        self.pose = np.zeros(3)


    def resurrect(self):
        self.is_alive = True


    def step(self, dt, external_force=0.0):
        self.ddpose = self.force / self.mass + external_force / self.mass

        dpose = self.dpose + self.ddpose * dt 
        self.dpose = dpose
        
        pose = self.pose + self.dpose * dt + 0.5 * self.ddpose * dt**2
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


    def __getitem__(self, index): 
        return self.points[index]
    

class Anchor:
    def __init__(self, init_pose, id=0):
        self.id = id
        self.pose = init_pose
        self.dpose = np.zeros(3)
        self.ddpose = np.zeros(3)
        self.trajectory = []
        self.traj_point = 0


    def set_trajectory(self, trajectory):
        self.trajectory = trajectory


    def step(self):
        self.pose = self.trajectory[self.traj_point]
        self.traj_point = (self.traj_point + 1) % len(self.trajectory)


class Agent_1D:
    def __init__(self, id=0, mass=1.0, x=0, is_alive=True, k_friction=0.01):
        self.id = id
        self.mass = mass
        self.pose = x
        self.dpose = 0
        self.ddpose = 0
        self.force = 0
        self.k_friction = k_friction

        self.has_peer = False
        self.peer_state = 0
        self.prior_knowledge = 0
        self.is_alive = is_alive

    
    def apply_force(self, force):
        self.force = force


    def set_dpose(self, dpose):
        self.dpose = dpose


    def kill(self):
        self.is_alive = False
        self.pose = 0


    def resurrect(self):
        self.is_alive = True


    def step(self, dt, gravity_force=0.0):
        self.ddpose = self.force / self.mass - self.k_friction * self.dpose - gravity_force

        dpose = self.dpose + self.ddpose * dt
        self.dpose = dpose
        
        pose = self.pose + self.dpose * dt + 0.5 * self.ddpose * dt**2
        self.pose = pose


class Pacemaker:
    def __init__(self, motion_func, x_0=0, velocity=1.0, dt=0.01):
        self.pose = x_0
        self.dpose = velocity
        self.dt = dt
        self.t = 0
        self.motion_func = motion_func

    
    def step(self):
        self.t += self.dt
        self.pose = self.motion_func(self.t)