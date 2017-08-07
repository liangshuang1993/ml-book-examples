import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
training_set = np.array([[(3, 3), 1], [(4, 3), 1], [(1, 1), -1]])
w = np.zeros(2)
b = 0
history = []
lr = 1
alpha = np.zeros(training_set.shape[0])

def init():
    for data in training_set:
        x, y = data[0]
        if data[1] == 1:
            plt.plot(x, y, 'bo')
        else:
            plt.plot(x, y, 'rx')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis([-6, 6, -6, 6])
    line.set_data([], [])
    return line

def check(i):
    global gram
    global labels
    tmp = np.dot(alpha * labels, gram[i])
    if((tmp + b) * labels[i] <= 0):
        return False
    return True

def update(alpha, b, i, data):
    alpha[i] += 1
    label = data[1]
    b += lr * label
    return (alpha, b)

def animate(i):
    global history, gram
    label = np.array(training_set[:, 1])
    w, b = history[i]
    print w, b
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

    # Gram matrix
    labels = training_set[:, 1]
    data_matrix = np.zeros((training_set.shape[0], 1, 2))
    for index, data in enumerate(training_set):
        data_matrix[index] = data[0]
    gram = data_matrix * np.transpose(data_matrix, (1, 0, 2))
    gram = np.sum(gram, axis=2)

    for i in range(100):
        flag = False
        for i, data in enumerate(training_set):
            if not check(i):
                flag = True
                alpha, b = update(alpha, b, i, data)   
                w[0] = np.dot(alpha * labels, data_matrix[:, :, 0])
                w[1] = np.dot(alpha * labels, data_matrix[:, :, 1])
                history.append([w, b])
        if not flag:
            break
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(history), interval=500, repeat=False)
    plt.show()