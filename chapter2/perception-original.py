import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

training_set = [[(3, 3), 1], [(4, 3), 1], [(1, 1), -1]]
w = [0, 0]
b = 0
history = []
lr = 1

def init():
    for data in training_set:
        x, y = data[0]
        if data[1] == 1:
            plt.plot(x, y, 'bo')
        else:
            plt.plot(x, y, 'rx')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis([-6, 6, -5, 5])
    line.set_data([], [])
    return line

def check(data):
    x, y = data[0]
    label = data[1]
    tmp = w[0] * x + w[1] * y + b
    if(tmp * label <= 0):
        return False
    return True

def update(w, b, data):
    x, y = data[0]
    label = data[1]
    w[0] += lr * label * x
    w[1] += lr * label * y
    b += lr * label
    return (w, b)

def animate(i):
    global history
    w, b = history[i]
    if w[0] == 0:
        x1 = -6
        y1 = -b / w[1]
        x2 = 6
        y2 = -b / w[1]
    elif w[1] == 0:
        y1 = -6
        x1 = -b / w[0]
        y2 = 6
        x2 = -b / w[0]
    else:
        x1 = -6
        y1 = -(b + w[0] * x1) / w[1]
        x2 = 6
        y2 = -(b + w[0] * x2) / w[1]

    line.set_data([x1, x2], [y1, y2])
    return line,

if __name__=='__main__':
    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes(xlim=(-6, 6), ylim=(-6, 6))
    line, =ax.plot([], [], lw=2)
    for i in range(100):
        flag = False
        for data in training_set:
            if not check(data):
                flag = True
                w, b = update(w, b, data)
                history.append([w, b])
        if not flag:
            break
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(history), interval=500, repeat=False)
    plt.show()