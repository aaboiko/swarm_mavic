import numpy as np
from swarming import SceneSupervisor

class PointMass:
    def __init__(self, id=0, family_id=0, mass=1.0, x=0, y=0, z=0, u_max=0.5, R_vis=1.0, is_alive=True):
        self.id=id
        self.family_id = family_id
        self.is_alive = is_alive

        self.mass = mass
        self.pose = np.array([x, y, z])
        self.dpose = np.zeros(3)
        self.ddpose = np.zeros(3)
        self.force = np.zeros(3)
        self.u_max = u_max

        self.has_peer = False
        self.peer_state = 0
        self.prior_knowledge = np.zeros(3)
        self.R_vis = R_vis


    def apply_force(self, force):
        self.force = force


    def set_dpose(self, dpose):
        self.dpose = dpose


    def step(self, dt):
        self.ddpose = self.force / self.mass

        dpose = self.dpose + self.ddpose * dt
        self.dpose = dpose
        
        pose = self.pose + self.dpose * dt + 0.5 * self.ddpose * dt**2
        self.pose = pose


    def get_nearest_distances(self, peers):
        min_dist_x_plus, min_dist_y_plus, min_dist_z_plus = np.inf, np.inf, np.inf
        min_dist_x_minus, min_dist_y_minus, min_dist_z_minus = np.inf, np.inf, np.inf
        id_x_plus, id_y_plus, id_z_plus = -1, -1, -1
        id_x_minus, id_y_minus, id_z_minus = -1, -1, -1

        for i in range(len(peers)):
            peer = peers[i]
            vx, vy, vz = peer

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
        mins = [min_dist_x_plus, min_dist_y_plus, min_dist_z_plus, min_dist_x_minus, min_dist_y_minus, min_dist_z_minus]
        
        return mins, ids
    

    def get_farthest_distances(self, peers):
        max_dist_x_plus, max_dist_y_plus, max_dist_z_plus = 0, 0, np.inf
        max_dist_x_minus, max_dist_y_minus, max_dist_z_minus = 0, 0, np.inf
        id_x_plus, id_y_plus, id_z_plus = -1, -1, -1
        id_x_minus, id_y_minus, id_z_minus = -1, -1, -1

        for i in range(len(peers)):
            peer = peers[i]
            vx, vy, vz = peer

            if vx >= 0:
                if vx > max_dist_x_plus:
                    max_dist_x_plus = vx
                    id_x_plus = i
            else:
                if -vx > max_dist_x_minus:
                    max_dist_x_minus = -vx
                    id_x_minus = i

            if vy >= 0:
                if vy > max_dist_y_plus:
                    max_dist_y_plus = vy
                    id_y_plus = i
            else:
                if -vy > max_dist_y_minus:
                    max_dist_y_minus = -vy
                    id_y_minus = i

            if vz >= 0:
                if vz < max_dist_z_plus:
                    max_dist_z_plus = vz
                    id_z_plus = i
            else:
                if -vz < max_dist_z_minus:
                    max_dist_z_minus = -vz
                    id_z_minus = i

        ids = [id_x_plus, id_y_plus, id_z_plus, id_x_minus, id_y_minus, id_z_minus]
        maxs = [max_dist_x_plus, max_dist_y_plus, max_dist_z_plus, max_dist_x_minus, max_dist_y_minus, max_dist_z_minus]
        
        return maxs, ids
    

    def xi(self, g):
        return np.tanh(g)
    

    def control_force(self, distances):
        d_x_plus, d_y_plus, d_z_plus, d_x_minus, d_y_minus, d_z_minus = distances
        vx, vy, vz = self.dpose

        sigma_x = vx - self.xi(d_x_plus) + self.xi(d_x_minus)
        sigma_y = vy - self.xi(d_y_plus) + self.xi(d_y_minus)
        sigma_z = vz - self.xi(d_z_plus) + self.xi(d_z_minus)

        ux = -self.u_max * np.sign(sigma_x)
        uy = -self.u_max * np.sign(sigma_y)
        uz = -self.u_max * np.sign(sigma_z)

        return np.array([ux, uy, uz])


class PointAgent(PointMass):
    def __init__(self, id=0, family_id=0, mass=1.0, x=0, y=0, z=0, u_max=1.0, R_vis=1.0, w=0.5):
        super().__init__(id=id, family_id=family_id, mass=mass, x=x, y=y, z=z, u_max=u_max, R_vis=R_vis)
        self.w = w


    def perform(self, peers, p_state, anchor_dir, dt):
        self.peer_state = p_state
        distances = []

        if len(peers) > 0:
            #mins, ids = self.get_nearest_distances(peers)
            mins, ids = self.get_farthest_distances(peers)

            for i in range(6):
                if ids[i] == -1:
                    if anchor_dir[i] < 0:
                        if i == 2 or i == 5:
                            distances.append(self.w)
                        else:
                            distances.append(0)
                    else:
                        distances.append(self.R_vis)
                else:
                    distances.append(mins[i])

        else:
            for i, item in enumerate(anchor_dir):
                if item >= 0:
                    distances.append(self.R_vis * item)
                else:
                    if i == 2 or i == 5:
                        distances.append(-self.w * item)
                    else:
                        distances.append(0)

        force = self.control_force(distances)
        self.apply_force(force)
        self.step(dt)


class Anchor(PointMass):
    def __init__(self, id=0, family_id=0, mass=1.0, x=0, y=0, z=0, u_max=0.5, R_vis=2.0, w=1.0):
        super().__init__(id=id, family_id=family_id, mass=mass, x=x, y=y, z=z, u_max=u_max, R_vis=R_vis)
        self.w = w


    def perform(self, peers, p_state, attraction_dir, dt):
        self.peer_state = p_state
        distances = []

        #diagonal grid
        if len(peers) > 0:
            mins, ids = self.get_nearest_distances(peers)

            for i in range(6):
                if ids[i] == -1:
                    if attraction_dir[i] < 0:
                        if i != 2 and i != 5:
                            distances.append(self.w)
                        else:
                            distances.append(0)
                    else:
                        distances.append(self.R_vis)
                else:
                    distances.append(mins[i])

        else:
            for i, item in enumerate(attraction_dir):
                if item >= 0:
                    distances.append(self.R_vis * item)
                else:
                    if i != 2 and i != 5:
                        distances.append(-self.w * item)
                    else:
                        distances.append(0)
        
        #parallel grid
        '''mins, ids = self.get_nearest_distances(peers)
        x_plus, y_plus, z_plus, x_minus, y_minus, z_minus = mins

        statements = [
            ((x_plus <= y_plus) or (x_plus <= y_minus)) * max(x_plus, max(y_plus, y_minus)),
            ((y_plus < x_plus) or (y_plus < x_minus)) * max(y_plus, max(x_plus, x_minus)),
            z_plus,
            ((x_minus <= y_plus) or (x_minus <= y_minus)) * max(x_minus, max(y_plus, y_minus)),
            ((y_minus < x_plus) or (y_minus < x_minus)) * max(y_minus, max(x_plus, x_minus)),
            z_minus
        ]

        for i in range(6):
            if ids[i] == -1:           #if no neighbors
                if attraction_dir[i] < 0: #if no neighbors and no attraction
                    if i != 2 and i != 5:
                        distances.append(self.w)
                    else:
                        distances.append(0)
                else:
                    distances.append(self.R_vis) #no neighbors but has attraction
            else:
                distances.append(statements[i])      #has neighbor'''

        force = self.control_force(distances)
        self.apply_force(force)
        self.step(dt)


class AnchorPredefined:
    def __init__(self, id=0, family_id=0, x=0, y=0, z=0, is_alive=True, trajectory=[]):
        self.id = id
        self.family_id = family_id
        self.is_alive = is_alive
        self.R_vis = 0

        self.pose = np.array([x, y, z])
        self.dpose = np.zeros(3)
        self.ddpose = np.zeros(3)
        self.trajectory = trajectory
        self.traj_point = 0


    def set_trajectory(self, trajectory):
        self.trajectory = trajectory


    def step(self):
        self.pose = self.trajectory[self.traj_point]
        self.traj_point = (self.traj_point + 1) % len(self.trajectory)


class AnchorStatic:
    def __init__(self, id=0, family_id=0, x=0, y=0, z=0, is_alive=True):
        self.id = id
        self.family_id = family_id
        self.R_vis = 0
        self.is_alive = is_alive
        self.pose = np.array([x, y, z])