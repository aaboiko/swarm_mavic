import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils import PointAgent, Anchor, AnchorPredefined, AnchorStatic
from matplotlib.patches import Circle
from mpl_toolkits import mplot3d
from tqdm import tqdm
from scene_creator import static_sin_points, chain_poses, spinning_line_points, sliding_line_points


n_params = 13

data = []

fig_3d = plt.figure(figsize = (20, 20))
ax_3d = fig_3d.add_subplot(projection='3d')

s_point = 30
s_anchor = 60

s_point_3d = 100
s_anchor_3d = 200

circle_colors = ["blue", "green", "red"]

#x_bounds = (-10, 10)
#y_bounds = (-10, 10)
#z_bounds = (0, 10)

x_bounds = (-6, 6)
y_bounds = (-6, 6)
z_bounds = (0, 6)

x_agent_min, x_agent_max = -10, 10
y_agent_min, y_agent_max = -10, 10
z_agent_min, z_agent_max = 4, 10

x_anchor_min, x_anchor_max = -10, 10
y_anchor_min, y_anchor_max = -10, 10
z_anchor_min, z_anchor_max = 1, 3

anchors_attraction_point = np.array([0, 0, 1])
family_colors = ["magenta", "blue", "violet", "black", "brown", "aquamarine", "aqua", "gold", "coral", "chocolate", "purple", "teal", "pink", "gold", "violet", "magenta"]


def get_peers(points, focus_point, attraction_point, is_anchor=False):
    peers = []
    peer_state = 0

    if not is_anchor:
        if np.linalg.norm(attraction_point - focus_point.pose) <= focus_point.R_vis:
            peer_state = 2
            vec = attraction_point - focus_point.pose
            peers.append(vec)

    for point in points:
        d = np.linalg.norm(point.pose - focus_point.pose)

        if (is_anchor and d <= point.R_vis and point.id != focus_point.id and point.is_alive) or (not is_anchor and d <= point.R_vis and point.id != focus_point.id and focus_point.family_id == point.family_id and point.is_alive):
            if peer_state == 0:
                peer_state = 1

            peer_state = max(peer_state, point.peer_state)
            vec = point.pose - focus_point.pose
            peers.append(vec)
    
    return peers, peer_state


def write_log(path, points):
    with open(path, "a") as file:
        line = f""

        for point in points:
            x, y, z = point.pose
            dx, dy, dz = point.dpose
            ux, uy, uz = point.force
            peer_state = int(point.peer_state)
            family_id = int(point.family_id)
            R_vis = point.R_vis
            is_alive = int(point.is_alive)

            line += f"{x} {y} {z} {dx} {dy} {dz} {ux} {uy} {uz} {peer_state} {family_id} {R_vis} {is_alive} "

        file.write(line + "\n")


def process(log_path_agents, 
            log_path_anchors,
            n_anchors,
            n_agents_in_family,
            dt=0.02,
            log_step=1,
            R_vis=2.5, 
            n_iters=10000,
            u_max=10.0,
            w=0.5,
            sigma=0.8,
            agents_initialization="chain",
            anchors_initialization="certain",
            predefined_anchor_traj_params=None,
            anchor_mode="processing"):
    
    #agents = [PointAgent(id=i, family_id=(i // N_AGENTS_IN_FAMILY), x=np.random.uniform(x_agent_min, x_agent_max), y=np.random.uniform(y_agent_min, y_agent_max), z=np.random.uniform(z_agent_min, z_agent_max), u_max=10.0) for i in range(N_ANCHORS * N_AGENTS_IN_FAMILY)]

    agents = []
    #anchor_points = static_sin_points()
    anchor_points = spinning_line_points(gap=1, n_agents=n_anchors)
    #anchor_points = sliding_line_points(n_agents=n_anchors)
    cur_id = 0

    for point in anchor_points:
        anch_x, anch_y, anch_z = point
        poses = chain_poses(n_agents_in_family, R_vis, 0.1, sigma, sigma, x_anchor=anch_x, y_anchor=anch_y, z_anchor=anch_z)

        for pose in poses:
            x, y, z = pose
            agent = PointAgent(id=cur_id, family_id=(cur_id // n_agents_in_family), x=x, y=y, z=z, R_vis=R_vis, u_max=u_max, w=w)
            agents.append(agent)
            cur_id += 1

    if anchor_mode == "processing":
        anchors = [Anchor(id=i, family_id=i, x=np.random.uniform(x_anchor_min, x_anchor_max), y=np.random.uniform(y_anchor_min, y_anchor_max), z=np.random.uniform(z_anchor_min, z_anchor_max)) for i in range(n_anchors)]
    if anchor_mode == "predefined":
        with open(log_path_anchors, "r") as file:
            trajs_lines = [[float(item) for item in traj_line.rstrip().split(' ')] for traj_line in file]
            logs = np.array(trajs_lines)
            anchors_logs = np.hsplit(logs, n_anchors)

        anchors = [AnchorPredefined(id=i, family_id=i, trajectory=traj[:,0:3]) for i, traj in enumerate(anchors_logs)]

    print("processing...")

    open(log_path_agents, "w").close()

    if anchor_mode == "processing":
        open(log_path_anchors, "w").close()

    for i in tqdm(range(n_iters)):
        if i % log_step == 0:
            write_log(log_path_agents, agents)

        if anchor_mode == "processing"and i % log_step == 0:
            write_log(log_path_anchors, anchors)

        for anchor in anchors:
            if anchor_mode == "processing":
                anchor_peers, anchor_p_state = get_peers(anchors, anchor, anchors_attraction_point, is_anchor=True)
                attraction_vec = anchors_attraction_point - anchor.pose
                attraction_x, attraction_y, attraction_z = attraction_vec
                anchor.prior_knowledge = np.array([np.sign(attraction_x), np.sign(attraction_y), np.sign(attraction_z)])
                attraction_dir = [np.sign(attraction_x), np.sign(attraction_y), np.sign(attraction_z), -np.sign(attraction_x), -np.sign(attraction_y), -np.sign(attraction_z)]

                anchor.perform(anchor_peers, anchor_p_state, attraction_dir, dt)

            if anchor_mode == "predefined":
                anchor.step()

        for agent in agents:
            anchor_pose = anchors[agent.family_id].pose
            peers, p_state = get_peers(agents, agent, anchor_pose)
            vec_to_anchor = anchor_pose - agent.pose
            x_to_anchor, y_to_anchor, z_to_anchor = vec_to_anchor
            agent.prior_knowledge = np.array([np.sign(x_to_anchor), np.sign(y_to_anchor), np.sign(z_to_anchor)])
            signs = [np.sign(x_to_anchor), np.sign(y_to_anchor), np.sign(z_to_anchor), -np.sign(x_to_anchor), -np.sign(y_to_anchor), -np.sign(z_to_anchor)]

            agent.perform(peers, p_state, signs, dt)


def animate_xy(i):
    plt.clf()

    plt.xlabel('x')
    plt.xlim(x_bounds)
    plt.ylim(y_bounds)
    plt.ylabel('y')

    plt.gca().set_aspect('equal', adjustable='box')
    nums_agents, nums_anchors = data[i]

    agents = np.array(nums_agents).reshape(-1, n_params)
    anchors = np.array(nums_anchors).reshape(-1, n_params)

    for agent in agents:
        x, y, z, dx, dy, dz, ux, uy, uz, peer_state, family_id, R_vis, is_alive = agent
        
        if is_alive:
            #circle_color = circle_colors[int(peer_state)]
            circle_color="grey"
            point_color = family_colors[int(family_id)]

            plt.scatter(x, y, s=s_point, color=point_color)
            circle = Circle(xy=(x, y), radius=R_vis, color=circle_color, alpha=0.2)
            plt.gca().add_artist(circle)

    for anchor in anchors:
        x, y, z, dx, dy, dz, ux, uy, uz, peer_state, family_id, R_vis, is_alive = anchor
        
        if is_alive:
            circle_color = circle_colors[int(peer_state)]
            point_color = family_colors[int(family_id)]

            plt.scatter(x, y, s=s_anchor, color="red")
            circle = Circle(xy=(x, y), radius=R_vis, color=circle_color, alpha=0.2)
            plt.gca().add_artist(circle)


def animate_xz(i):
    plt.clf()

    plt.xlabel('x')
    plt.xlim(x_bounds)
    plt.ylim(z_bounds)
    plt.ylabel('z')

    plt.gca().set_aspect('equal', adjustable='box')
    nums_agents, nums_anchors = data[i]

    agents = np.array(nums_agents).reshape(-1, n_params)
    anchors = np.array(nums_anchors).reshape(-1, n_params)

    for agent in agents:
        x, y, z, dx, dy, dz, ux, uy, uz, peer_state, family_id, R_vis, is_alive = agent
        
        if is_alive:
            circle_color = circle_colors[int(peer_state)]
            point_color = family_colors[int(family_id)]

            plt.scatter(x, z, s=s_point, color=point_color)
            circle = Circle(xy=(x, z), radius=R_vis, color=circle_color, alpha=0.2)
            plt.gca().add_artist(circle)

    for anchor in anchors:
        x, y, z, dx, dy, dz, ux, uy, uz, peer_state, family_id, R_vis, is_alive = anchor
        
        if is_alive:
            circle_color = circle_colors[int(peer_state)]
            point_color = family_colors[int(family_id)]

            plt.scatter(x, z, s=s_anchor, color="red")
            circle = Circle(xy=(x, z), radius=R_vis, color=circle_color, alpha=0.2)
            plt.gca().add_artist(circle)


def create_gif(log_path_agents, log_path_anchors, gif_path, flag="xy", frame_step=20, start_frame=0, end_frame=10000, n_frames=10000):
    print(f"creating gif in {flag} projection...")
    iters = 0

    with open(log_path_agents, "r") as f_agents, open(log_path_anchors, "r") as f_anchors:
        agents_lines = [agents_line for agents_line in f_agents]
        anchors_lines = [anchors_line for anchors_line in f_anchors]

        for line_agents, line_anchors in zip(agents_lines, anchors_lines):
            nums_agents = [float(item) for item in line_agents.rstrip().split(' ')]
            nums_anchors = [float(item) for item in line_anchors.rstrip().split(' ')]

            data.append((nums_agents, nums_anchors))
            iters += 1

            if iters > n_frames:
                 break
            
        if flag != "3d":
            fig, ax = plt.subplots()

        if flag == "xy":
            anim = animation.FuncAnimation(fig, animate_xy, frames = tqdm(range(start_frame, end_frame, frame_step)), interval = 20)
        if flag == "xz":
            anim = animation.FuncAnimation(fig, animate_xz, frames = tqdm(range(start_frame, end_frame, frame_step)), interval = 20)
        if flag == "3d":
            anim = animation.FuncAnimation(fig_3d, animate_3d, frames = tqdm(range(start_frame, end_frame, frame_step)), interval = 20)

        anim.save(gif_path, fps = 60, writer = 'pillow')


def animate_3d(i):
    ax_3d.clear()

    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('z')

    ax_3d.set_xlim(x_bounds)
    ax_3d.set_ylim(y_bounds)
    ax_3d.set_zlim(z_bounds)

    ax_3d.set_aspect('equal', adjustable='box')
    nums_agents, nums_anchors = data[i]

    agents = np.array(nums_agents).reshape(-1, n_params)
    anchors = np.array(nums_anchors).reshape(-1, n_params)

    for agent in agents:
        x, y, z, dx, dy, dz, ux, uy, uz, peer_state, family_id, R_vis, is_alive = agent
        
        if is_alive:
            point_color = family_colors[int(family_id)]
            ax_3d.scatter(x, y, z, s=s_point_3d, color=point_color)

    for anchor in anchors:
        x, y, z, dx, dy, dz, ux, uy, uz, peer_state, family_id, R_vis, is_alive = anchor
        
        if is_alive:
            point_color = family_colors[int(family_id)]
            ax_3d.scatter(x, y, z, s=s_anchor_3d, color="red")


def launch(scene_name, 
           n_anchors,
           n_agents_in_family,
           dt=0.02,
           log_step=1,
           n_iters=10000,
           u_max=10.0,
           sigma=0.8,
           anchor_mode="processing",
           agents_log_folder="grid_antenna/logs/agents", 
           anchors_log_folder="grid_antenna/logs/anchors", 
           anchors_predefined_log_folder="grid_antenna/logs/anchors_predefined", 
           suffix=""):
    
    if anchor_mode not in ["processing", "predefined", "static"]:
        print("incorrect anchor mode")
        return
    
    if len(suffix) > 0:
        suffix = f"_{suffix}"
    
    log_path_agents = f"{agents_log_folder}/log_agents_{scene_name}{suffix}.txt"

    if anchor_mode == "processing":
        log_path_anchors = f"{anchors_log_folder}/log_anchors_{scene_name}{suffix}.txt"
    if anchor_mode == "predefined":
        log_path_anchors = f"{anchors_predefined_log_folder}/{scene_name}.txt"

    gif_path_xy = f"grid_antenna/gifs/{scene_name}{suffix}_xy.gif"
    gif_path_3d = f"grid_antenna/gifs/{scene_name}{suffix}_3d.gif"
    gif_path_3d_end = f"grid_antenna/gifs/{scene_name}{suffix}_3d_part_2.gif"

    #process(log_path_agents, log_path_anchors, n_anchors, n_agents_in_family, anchor_mode=anchor_mode, u_max=u_max, dt=dt, log_step=log_step, n_iters=n_iters, sigma=sigma)

    #create_gif(log_path_agents, log_path_anchors, gif_path_xy, flag="xy", frame_step=20)
    create_gif(log_path_agents, log_path_anchors, gif_path_3d, flag="3d", frame_step=45)
    
    #create_gif(log_path_agents, log_path_anchors, gif_path_3d, flag="3d", frame_step=20, start_frame=0, end_frame=4000)


scene_name = "spinning_line"

launch(scene_name, 6, 5, anchor_mode="predefined", dt=0.02, log_step=1, n_iters=10000, u_max=15.0, sigma=0.6)