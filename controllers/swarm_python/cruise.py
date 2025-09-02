import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle

from point_mass import Agent_1D, Pacemaker
from matplotlib.patches import Circle
from tqdm import tqdm
from functools import partial
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

n_params = 6
gravity = 0.981

R_vis = 1.0
w = 0.5
u_max = 20.0

data = []

s_point = 30
s_anchor = 60
s_point_3d = 100
s_anchor_3d = 200

fig_3d = plt.figure(figsize = (20, 20))
ax_3d = fig_3d.add_subplot(projection='3d')

x_bounds = (-9, 9)
y_bounds = (-9, 9)
z_bounds = (-2, 2)

colorize_perception_area = True
points_colors = ["magenta", "blue", "violet", "black", "brown", "aquamarine", "aqua", "gold", "coral", "chocolate", "purple", "teal", "pink", "gold", "violet", "magenta"]
circle_colors = ["blue", "green", "red"]

pacemaker_velocity = -0.1
pacemaker_acc = -0.25

integrand = lambda t: np.sqrt(64 + 9 * np.cos(3 * t)**2)
pacemaker_motion_func = lambda t: -0.15 * t + (0.5 / np.pi) * np.cos(2 * np.pi / 10)
#pacemaker_motion_func = lambda t: pacemaker_velocity * t 
spatial_traj_func = lambda s: np.array([8 * np.cos(s), 8 * np.sin(s), np.sin(3 * s)])
spatial_traj_derivative_func = lambda s: np.array([-8 * np.sin(s), 8 * np.cos(s), 3 * np.cos(3 * s)])


def forward_integrate(t):
    res, _ = integrate.quad(integrand, 0, t)
    return res


def reverse_integrate(s, t_guess=1.0):
    sol = root_scalar(lambda t: forward_integrate(t) - s, 
                      bracket=[0, 2 * np.pi],  
                      method='brentq')  
    return sol.root


#t_values = np.linspace(-6 * np.pi, 6 * np.pi, 200000)
#s_values = np.array([forward_integrate(t) for t in t_values])
#t_interpolator = interp1d(s_values, t_values, kind='cubic', fill_value="extrapolate")
#pickle.dump(t_interpolator, open("t_interpolator.bin", "wb"))
t_interpolator = pickle.load(open("t_interpolator.bin", "rb"))


def sign(x):
    if x >= 0:
        return 1
    
    return -1


def xi(g):
    return np.tanh(g)


def get_peers(points, focus_point, anchor_pose):
    peers = []
    peer_state = 0
    has_peer_forward = False

    if abs(anchor_pose - focus_point.pose) <= R_vis:
        peer_state = 2
        vec = anchor_pose - focus_point.pose
        peers.append(vec)
        has_peer_forward = True

    for point in points:
        d = abs(point.pose - focus_point.pose) + np.random.normal(0, 0.0001)

    for point in points:
        d = np.linalg.norm(point.pose - focus_point.pose) + np.random.normal(0, 0.0001)

        if d <= R_vis and point.id != focus_point.id and point.is_alive:
            if peer_state == 0:
                peer_state = 1

            peer_state = max(peer_state, point.peer_state)
            vec = point.pose - focus_point.pose
            peers.append(vec)

            if point.pose >= focus_point.pose:
                has_peer_forward = True
    
    return peers, peer_state, has_peer_forward


def get_nearest_distances(peers):
    min_dist_plus, min_dist_minus = np.inf, np.inf
    id_plus, id_minus = -1, -1

    for i in range(len(peers)):
        v = peers[i]

        if v >= 0:
            if v < min_dist_plus:
                min_dist_plus = v
                id_plus = i
        else:
            if -v < min_dist_minus:
                min_dist_minus = -v
                id_minus = i

    has_peer_forward = id_plus > -1
    has_peer_back = id_minus > -1

    return min_dist_plus, min_dist_minus, has_peer_forward, has_peer_back


def g(min_dist, gap, has_peer):
    return min_dist if has_peer else gap


def g_gothic(min_dist, has_peer):
    return min(g(min_dist, R_vis, has_peer), w)


def control_force(min_dist_plus, min_dist_minus, has_peer_forward, has_peer_back, point):
    dx = point.dpose
    sigma = dx - xi(g_gothic(min_dist_plus, has_peer_forward)) + xi(g(min_dist_minus, R_vis, has_peer_back))
    u = -u_max * sign(sigma)

    return u


def write_log(path, points):
    with open(path, "a") as file:
        line = f""

        for point in points:
            x = point.pose
            dx = point.dpose
            u = point.force
            peer_state = int(point.peer_state)
            id = point.id
            is_alive = int(point.is_alive)

            line += f"{x} {dx} {u} {peer_state} {id} {is_alive} "

        file.write(line + "\n")


def process(n_points,
            n_iters,
            dt,
            log_path,
            pacemaker_motion_func,
            spatial_traj_func,
            spatial_traj_derivative_func
            ):
    
    print('processing...')

    R_min = 0.2
    points = []
    cur_x = 0

    for i in range(n_points):
        interval = np.random.uniform(R_min, 2*R_vis)
        x = cur_x + interval
        point = Agent_1D(id=i, x=x)
        points.append(point)
        cur_x = x

    pacemaker = Pacemaker(pacemaker_motion_func, dt=dt)

    open(log_path, "w").close()

    for iter in tqdm(range(n_iters)):
        write_log(log_path, points)

        for point in points:
            if point.is_alive:
                peers, p_state, has_peer_forward = get_peers(points, point, pacemaker.pose)
                point.peer_state = p_state

                min_dist_plus, min_dist_minus, has_peer_forward, has_peer_back = get_nearest_distances(peers)
                t_of_traj = t_interpolator(point.pose)
                dot_x, dot_y, dot_z = spatial_traj_derivative_func(t_of_traj)
                sin_phi = dot_z / np.sqrt(dot_x**2 + dot_y**2 + dot_z**2)
                gravity_force = gravity * sin_phi

                force = control_force(min_dist_plus, min_dist_minus, has_peer_forward, has_peer_back, point) 
                point.apply_force(force)
                point.step(dt, gravity_force=gravity_force)

        pacemaker.step()


def circle_transform(pose, radius=8.0):
    phi = pose / radius

    return np.array([
        radius * np.cos(phi),
        radius * np.sin(phi)
    ])


def animate(i, spatial_traj_func=spatial_traj_func, dt=0.01):
    nums = data[i]
    points = np.array(nums).reshape(-1, n_params)
    pacemaker_pose = pacemaker_motion_func(i * dt)
    circle_colors = ["blue", "green", "red"]
    plt.clf()

    plt.xlabel('x')
    plt.xlim((-9, 9))
    plt.ylim((-9, 9))
    plt.ylabel('y')

    plt.gca().set_aspect('equal', adjustable='box')
    traj_circle = Circle(xy=(0, 0), radius=8.0, color="green", fill=False)
    plt.gca().add_artist(traj_circle)

    x_pacemaker, y_pacemaker, z_pacemaker = spatial_traj_func(pacemaker_pose)
    plt.scatter(x_pacemaker, y_pacemaker, s=s_anchor, color="red")

    for point in points:
        pose, dpose, u, peer_state, id, is_alive = point
        
        if int(is_alive) > 0:
            if colorize_perception_area:
                circle_color = circle_colors[int(peer_state)]
            else:
                circle_color = "grey"

            x, y, z = spatial_traj_func(pose)
            plt.scatter(x, y, s=s_point, color="blue")
            circle = Circle(xy=(x, y), radius=R_vis, color=circle_color, alpha=0.2)

            plt.gca().add_artist(circle)


def animate_3d(i, spatial_traj_func=spatial_traj_func, pacemaker_motion_func=pacemaker_motion_func, dt=0.01):
    ax_3d.clear()

    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('z')

    ax_3d.set_xlim(x_bounds)
    ax_3d.set_ylim(y_bounds)
    ax_3d.set_zlim(z_bounds)

    ax_3d.set_aspect('equal', adjustable='box')
    nums = data[i]
    t = i * dt

    traj_points = np.array([spatial_traj_func(s) for s in np.linspace(0, 2 * np.pi, 100)])
    xs = traj_points[:,0]
    ys = traj_points[:,1]
    zs = traj_points[:,2]
    ax_3d.plot(xs, ys, zs, color="green")

    points = np.array(nums).reshape(-1, n_params)
    pacemaker_pose = pacemaker_motion_func(t)
    pacemaker_t_param = t_interpolator(pacemaker_pose)
    x_pacemaker, y_pacemaker, z_pacemaker = spatial_traj_func(pacemaker_t_param)
    ax_3d.scatter(x_pacemaker, y_pacemaker, z_pacemaker, s=s_anchor_3d, color="red")

    for point in points:
        pose, dpose, u, peer_state, id, is_alive = point

        if int(is_alive) > 0:
            point_t_param = t_interpolator(pose)
            x, y, z = spatial_traj_func(point_t_param)
            ax_3d.scatter(x, y, z, s=s_point_3d, color="blue")


def create_gif(log_path, gif_path, spatial_traj_func, pacemaker_motion_func, dt, frame_step=4, flag="xy", n_frames=10000):
    print(f"creating gif {gif_path}")
    iters = 0

    with open(log_path, "r") as file:
        lines = [line for line in file]

        for line in lines:
            nums = [float(item) for item in line.rstrip().split(' ')]
            data.append(nums)
            iters += 1

            if iters >= n_frames:
                break
        
        if flag == "xy":
            fig, ax = plt.subplots()
            anim = animation.FuncAnimation(fig, partial(animate, spatial_traj_func=spatial_traj_func, dt=dt), frames = tqdm(range(0, n_frames, frame_step)), interval = 20)
        if flag == "3d":
            anim = animation.FuncAnimation(fig_3d, partial(animate_3d, spatial_traj_func=spatial_traj_func, pacemaker_motion_func=pacemaker_motion_func, dt=dt), frames = tqdm(range(0, n_frames, frame_step)), interval = 20)

        anim.save(gif_path, fps = 60, writer = 'pillow')


def plot_graph(log_path, graph_path, dt=0.01):
    print(f"plotting graph: {graph_path}")

    with open(log_path, "r") as file:
        line_num = 0
        t_prev, zs_prev, alives_prev = 0, [], []

        fig = plt.figure(figsize = (20, 20))
        ax = fig.add_subplot()

        for line in tqdm(file):
            t = line_num * dt

            nums = [float(item) for item in line.rstrip().split(' ')]
            points = np.array(nums).reshape(-1, n_params)
            pacemaker_pose = pacemaker_motion_func(t)

            dists, alives = [], []

            for point in points:
                pose, dpose, u, peer_state, id, is_alive = point
                dist = pacemaker_pose - pose
                #print(f"pose: {pose}, pacemaker_pose: {pacemaker_pose}, dist: {dist}")
                dists.append(dist)
                alives.append(int(is_alive))

            if line_num > 0:
                for i, (z_diff, z_prev, alive, alive_prev) in enumerate(zip(dists, zs_prev, alives, alives_prev)):
                    if alive > 0 and alive_prev > 0:
                        ax.plot((t_prev, t), (z_prev, z_diff), color=points_colors[i])

            zs_prev = dists
            t_prev = t
            alives_prev = alives

            line_num += 1

        plt.xlabel('t', fontsize=20)
        plt.ylabel(r'$x_i - x_{pacemaker}$', fontsize=20)
        plt.title(r'$x_i - x_{pacemaker}$ for all agents', fontsize=20)
        #plt.ylim(z_bounds)
        
        plt.savefig(graph_path)


def run(scene_name, 
        n_points, 
        spatial_traj_func, 
        spatial_traj_derivative_func, 
        pacemaker_motion_func, 
        n_iters, 
        dt):
    
    log_path = f"logs/cruise_control/log_{scene_name}.txt"
    gif_path_xy = f"gifs/{scene_name}_xy.gif"
    gif_path_3d = f"gifs/{scene_name}_3d.gif"
    graph_path = f"gifs/{scene_name}.jpg"

    process(n_points,
            n_iters,
            dt, 
            log_path, 
            pacemaker_motion_func,
            spatial_traj_func,
            spatial_traj_derivative_func)
    
    create_gif(log_path, 
               gif_path_xy, 
               spatial_traj_func, 
               pacemaker_motion_func, 
               dt, 
               frame_step=10,
               n_frames=n_iters,
               flag="xy")
    
    create_gif(log_path, 
               gif_path_3d, 
               spatial_traj_func, 
               pacemaker_motion_func, 
               dt, 
               frame_step=25,
               n_frames=n_iters,
               flag="3d")

    plot_graph(log_path, graph_path, dt=dt)

run("cruise_control_spatial_var_10", 
    10, 
    spatial_traj_func,
    spatial_traj_derivative_func,
    pacemaker_motion_func,
    20000,
    0.01)

