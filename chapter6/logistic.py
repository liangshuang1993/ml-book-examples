import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

f = open('data.txt', 'r')

X = np.zeros([100, 3])
Y = np.zeros(100)

for i, line in enumerate(f.readlines()):
    X[i][0] = line.split()[0]
    X[i][1] = line.split()[1]
    X[i][2] = 1
    Y[i] = line.split()[2]

def gradient_up(iter_num, alpha):
    global X, Y
    w = np.zeros([3, 1])
    history = []

    for i in range(iter_num):
        tmp = np.array(np.mat(X) * np.mat(w))
        p = np.e ** tmp / (1 + np.e ** tmp)
        Y = Y.reshape([100, 1])
        gradient = alpha * np.mat(X.transpose()) * np.mat((Y - p))
        w += gradient
        history.append(w.copy())
    return history


def init():
    for i, label in enumerate(Y):
        if label == 1:
            plt.plot(X[i][0], X[i][1], 'ro')
        else:
            plt.plot(X[i][0], X[i][1], 'go')
    plt.xlabel('x0')
    plt.ylabel('x1')
    figure_line.set_data([], [])
    return figure_line


def animate(i):
    w = history[i]
    if w[0] == 0:
        x0 = -6
        x1 = 6
        y0 = - w[2] / w[1]
        y1 = - w[2] / w[1]
    elif w[1] == 0:
        y0 = -3
        y0 = 15
        x0 = - w[2] / w[0]
        x1 = - w[2] / w[0]
    else:
        x0 = -6
        y0 = -(w[2] + w[0] * x0) / w[1]
        x1 = 6
        y1 = -(w[2] + w[0] * x1) / w[1]
    figure_line.set_data([x0, x1], [y0, y1])
    return figure_line,

history = gradient_up(100, 0.1)
fig = plt.figure()
ax = plt.axes(xlim=(-6, 6), ylim=(-3, 15))
figure_line, =ax.plot([], [], lw=2)
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(history), interval=500, repeat=False)
plt.show()