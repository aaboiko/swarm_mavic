import numpy as np


class SceneSupervisor:
    def __init__(self):
        pass


    def get_mins_ids(self, peers):
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
    

    def get_nearest_distances(self, peers, anchor_dir, w, R_vis):
        ids, mins = self.get_mins_ids(peers)
        distances = []

        for i in range(6):
            if ids[i] == -1:
                if anchor_dir[i] < 0:
                    if i == 2 or i == 5:
                        distances.append(w)
                    else:
                        distances.append(0)
                else:
                    distances.append(R_vis)
            else:
                distances.append(mins[i])
        
        return distances
    

class DistrSwarmer:
    def __init__(self, w=0.5, R_vis=1.0):
        self.w = w
        self.R_vis = R_vis


    def xi(self, g):
        return np.tanh(g)


    def control_force(self, distances, agent):
        d_x_plus, d_y_plus, d_z_plus, d_x_minus, d_y_minus, d_z_minus = distances
        vx, vy, vz = agent.dpose

        sigma_x = vx - self.xi(d_x_plus) + self.xi(d_x_minus)
        sigma_y = vy - self.xi(d_y_plus) + self.xi(d_y_minus)
        sigma_z = vz - self.xi(d_z_plus) + self.xi(d_z_minus)

        ux = -self.u_max * np.sign(sigma_x)
        uy = -self.u_max * np.sign(sigma_y)
        uz = -self.u_max * np.sign(sigma_z)

        return np.array([ux, uy, uz])