import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils import PointAgent, Anchor
from matplotlib.patches import Circle

N_AGENTS_IN_FAMILY = 2
N_ANCHORS = 16

log_path_agents = "grid_antenna/logs/agents/log_agents_7.txt"
log_path_anchors = "grid_antenna/logs/anchors/log_anchors_7.txt"
gif_path = "grid_antenna/gifs/grid_antenna_7_xz.gif"

dt = 0.02
n_frames = 3000
data = []

s_point = 10
s_anchor = 20

x_bounds = (-10, 10)
y_bounds = (-10, 10)
z_bounds = (0, 20)

anchors_attraction_point = np.array([0, 0, 0])
family_colors = ["green", "blue", "orange", "black", "brown", "aquamarine", "aqua", "azure", "coral", "chocolate", "purple", "teal", "pink", "gold", "violet", "magenta"]


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

        if (is_anchor and d <= point.R_vis and point.id != focus_point.id) or (not is_anchor and d <= point.R_vis and point.id != focus_point.id and focus_point.family_id == point.family_id):
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

            line += f"{x} {y} {z} {dx} {dy} {dz} {ux} {uy} {uz} {peer_state} {family_id} {R_vis} "

        file.write(line + "\n")
        #print(line)


def process():
    anchors = [Anchor(id=i, family_id=i, x=np.random.uniform(x_bounds[0], x_bounds[1]), y=np.random.uniform(y_bounds[0], y_bounds[1]), z=np.random.uniform(z_bounds[0], z_bounds[1])) for i in range(N_ANCHORS)]
    agents = [PointAgent(id=i, family_id=(i // N_AGENTS_IN_FAMILY), x=np.random.uniform(x_bounds[0], x_bounds[1]), y=np.random.uniform(y_bounds[0], y_bounds[1]), z=np.random.uniform(z_bounds[0], z_bounds[1])) for i in range(N_ANCHORS * N_AGENTS_IN_FAMILY)]

    n_iters = 10000

    while n_iters > 0:
        n_iters -= 1
        print(f"iters left: {n_iters}")

        write_log(log_path_agents, agents)
        write_log(log_path_anchors, anchors)

        for anchor in anchors:
            anchor_peers, anchor_p_state = get_peers(anchors, anchor, anchors_attraction_point, is_anchor=True)
            attraction_vec = anchors_attraction_point - anchor.pose
            attraction_x, attraction_y, attraction_z = attraction_vec
            anchor.prior_knowledge = np.array([np.sign(attraction_x), np.sign(attraction_y), np.sign(attraction_z)])
            attraction_dir = [np.sign(attraction_x), np.sign(attraction_y), np.sign(attraction_z), -np.sign(attraction_x), -np.sign(attraction_y), -np.sign(attraction_z)]

            anchor.perform(anchor_peers, anchor_p_state, attraction_dir, dt)

        for agent in agents:
            anchor_pose = anchors[agent.family_id].pose
            peers, p_state = get_peers(agents, agent, anchor_pose)
            vec_to_anchor = anchor_pose - agent.pose
            x_to_anchor, y_to_anchor, z_to_anchor = vec_to_anchor
            agent.prior_knowledge = np.array([np.sign(x_to_anchor), np.sign(y_to_anchor), np.sign(z_to_anchor)])
            signs = [np.sign(x_to_anchor), np.sign(y_to_anchor), np.sign(z_to_anchor), -np.sign(x_to_anchor), -np.sign(y_to_anchor), -np.sign(z_to_anchor)]

            agent.perform(peers, p_state, signs, dt)


def animate_from_log(log_path_agents, log_path_anchors, axes="xy"):
    with open(log_path_agents, "r") as f_agents, open(log_path_anchors, "r") as f_anchors:
        line_num = 0
        agents_lines = [agents_line for agents_line in f_agents]
        anchors_lines = [anchors_line for anchors_line in f_anchors]

        for line_agents, line_anchors in zip(agents_lines, anchors_lines):
            line_num += 1
            print(f"line number: {line_num}")
            circle_colors = ["blue", "green", "red"]
            plt.clf()

            plt.xlabel('x')

            if axes == "xy":
                plt.xlim(x_bounds)
                plt.ylim(y_bounds)
                plt.ylabel('y')
            if axes == "xz":
                plt.xlim(x_bounds)
                plt.ylim(z_bounds)
                plt.ylabel('z')

            plt.gca().set_aspect('equal', adjustable='box')

            nums_agents = [float(item) for item in line_agents.rstrip().split(' ')]
            nums_anchors = [float(item) for item in line_anchors.rstrip().split(' ')]

            agents = np.array(nums_agents).reshape(-1, 12)
            anchors = np.array(nums_anchors).reshape(-1, 12)

            for agent in agents:
                x, y, z, dx, dy, dz, ux, uy, uz, peer_state, family_id, R_vis = agent
                circle_color = circle_colors[int(peer_state)]
                point_color = family_colors[int(family_id)]

                if axes == "xy":
                    plt.scatter(x, y, s=s_point, color=point_color)
                    circle = Circle(xy=(x, y), radius=R_vis, color=circle_color, alpha=0.2)
                if axes == "xz":
                    plt.scatter(x, z, s=s_point, color=point_color)
                    circle = Circle(xy=(x, z), radius=R_vis, color=circle_color, alpha=0.2)

                plt.gca().add_artist(circle)

            for anchor in anchors:
                x, y, z, dx, dy, dz, ux, uy, uz, peer_state, family_id, R_vis = anchor
                circle_color = circle_colors[int(peer_state)]
                point_color = family_colors[int(family_id)]

                if axes == "xy":
                    plt.scatter(x, y, s=s_anchor, color="red")
                    circle = Circle(xy=(x, y), radius=R_vis, color=circle_color, alpha=0.2)
                if axes == "xz":
                    plt.scatter(x, z, s=s_anchor, color="red")
                    circle = Circle(xy=(x, z), radius=R_vis, color=circle_color, alpha=0.2)

                plt.gca().add_artist(circle)

            plt.pause(0.01)

        plt.show()


def animate_xy(i):
    print(f"animate: {i}/{n_frames}")

    circle_colors = ["blue", "green", "red"]
    plt.clf()

    plt.xlabel('x')
    plt.xlim(x_bounds)
    plt.ylim(y_bounds)
    plt.ylabel('y')

    plt.gca().set_aspect('equal', adjustable='box')
    nums_agents, nums_anchors = data[i]

    agents = np.array(nums_agents).reshape(-1, 12)
    anchors = np.array(nums_anchors).reshape(-1, 12)

    for agent in agents:
        x, y, z, dx, dy, dz, ux, uy, uz, peer_state, family_id, R_vis = agent
        circle_color = circle_colors[int(peer_state)]
        point_color = family_colors[int(family_id)]

        plt.scatter(x, y, s=s_point, color=point_color)
        circle = Circle(xy=(x, y), radius=R_vis, color=circle_color, alpha=0.2)
        plt.gca().add_artist(circle)

    for anchor in anchors:
        x, y, z, dx, dy, dz, ux, uy, uz, peer_state, family_id, R_vis = anchor
        circle_color = circle_colors[int(peer_state)]
        point_color = family_colors[int(family_id)]

        plt.scatter(x, y, s=s_anchor, color="red")
        circle = Circle(xy=(x, y), radius=R_vis, color=circle_color, alpha=0.2)
        plt.gca().add_artist(circle)


def animate_xz(i):
    print(f"animate: {i}/{n_frames}")

    circle_colors = ["blue", "green", "red"]
    plt.clf()

    plt.xlabel('x')
    plt.xlim(x_bounds)
    plt.ylim(z_bounds)
    plt.ylabel('z')

    plt.gca().set_aspect('equal', adjustable='box')
    nums_agents, nums_anchors = data[i]

    agents = np.array(nums_agents).reshape(-1, 12)
    anchors = np.array(nums_anchors).reshape(-1, 12)

    for agent in agents:
        x, y, z, dx, dy, dz, ux, uy, uz, peer_state, family_id, R_vis = agent
        circle_color = circle_colors[int(peer_state)]
        point_color = family_colors[int(family_id)]

        plt.scatter(x, z, s=s_point, color=point_color)
        circle = Circle(xy=(x, z), radius=R_vis, color=circle_color, alpha=0.2)
        plt.gca().add_artist(circle)

    for anchor in anchors:
        x, y, z, dx, dy, dz, ux, uy, uz, peer_state, family_id, R_vis = anchor
        circle_color = circle_colors[int(peer_state)]
        point_color = family_colors[int(family_id)]

        plt.scatter(x, z, s=s_anchor, color="red")
        circle = Circle(xy=(x, z), radius=R_vis, color=circle_color, alpha=0.2)
        plt.gca().add_artist(circle)


def create_gif(log_path_agents, log_path_anchors, gif_path):
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
            
        fig, ax = plt.subplots()
        anim = animation.FuncAnimation(fig, animate_xz,  frames = len(data), interval = len(data))
        anim.save(gif_path, fps = 60, writer = 'pillow')


#process()
#animate_from_log(log_path_agents, log_path_anchors, axes="xy")
create_gif(log_path_agents, log_path_anchors, gif_path)