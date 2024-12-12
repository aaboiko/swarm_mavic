import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

x_bound_min = -10
x_bound_max = 10
y_bound_min = -10
y_bound_max = 10
z_bound_min = -1
z_bound_max = 10

s_point = 10
s_anchor = 20

data = []
fig, ax = plt.subplots()
x_anchor = np.array([0.0, 0.0, 0.0])   

with open("logs/log_1.txt", "r") as file:
    for line in file:
        nums = [float(item) for item in line.rstrip().split(' ')]
        
        data.append(nums)


def animate(i):
    nums = data[i]
    plt.clf()

    plt.xlim((y_bound_min, y_bound_max))
    plt.ylim((z_bound_min, z_bound_max))

    x_anc, y_anc, z_anc = x_anchor
    plt.scatter(x_anc, z_anc, s=s_anchor, color="red")

    for i in np.arange(0, len(nums), 3):
            x = nums[i]
            y = nums[i + 1]
            z = nums[i + 2]

            plt.scatter(x, z, s=10, color='blue')

# создаем анимацию
anim = animation.FuncAnimation(fig, animate,  frames = len(data), interval = len(data))
# сохраняем анимацию в формате GIF в папку со скриптом
anim.save(os.getcwd() + '/anim_var2.gif', fps = 5, writer = 'pillow')
