import numpy as np
import matplotlib.pyplot as plt

f = open('data1.txt', 'r')

x = np.mat(np.zeros([100, 2]), dtype=float)
y = np.mat(np.zeros([100, 1]), dtype=float)
for i, line in enumerate(f.readlines()):
    x[i, 0] = line.split()[0]
    x[i, 1] = line.split()[1]
    y[i, 0] = line.split()[2]


class SVM(object):
    def __init__(self):
        pass

    @staticmethod
    def get_kernel_matrix(x, option):
        m = x.shape[0]
        value = np.mat(np.zeros([m, m]))
        if option[0] == 'linear':
            print 'test'
            for i in range(m):
                x_i = x[i]
                value[:, i] = x * x_i.T
        elif option[0] == 'gussian':
            sigma = option[1]
            for i in range(m):
                x_i = x[i]
                for j in range(m):
                    delta = x[j] - x_i
                    value[i, j] = delta * delta.T
            value = np.exp(-value / (2.0 * sigma ** 2))
        return value


class SMO(object):
    def __init__(self, x, y, C):
        self.alphas = np.mat(np.zeros([100, 1]))
        self.x = x
        self.y = y
        self.C = C
        self.b = 0
        self.kernel_matrix = SVM().get_kernel_matrix(self.x, ['linear'])

    def check_kkt(self, i):
        """
        check if i-th alpha satisify kkt condition
        :param i: index of alpha
        :return: True if satisify
        """
        alpha = self.alphas[i, 0]
        e_i = self.get_error(i)

        if (alpha == 0.0) & ((y[i, 0] * e_i) < 0):
            return False
        elif alpha == self.C & y[i, 0] * e_i > 0:
            return False
        else:
            return True


    def get_error(self, i):
        """
        get error for index i, change with alpha and b
        :param i: index
        :return:float type value
        """
        g_i = (self.kernel_matrix[i] * np.multiply(self.alphas, self.y)) + self.b
        return g_i[0, 0] - self.y[i, 0]

    def select_first_alpha(self):
        """
        traversal all alphas, find a alpha doesn't satisify KKT condition, if can't find alpha, then return False
        :return: return alpha or False
        """
        m = self.x.shape[0]
        for i in range(m):
            if not self.check_kkt(i):
                return i
        return None

    def select_second_alpha(self, i, error_i):
        """
        find second alpha, inner loop
        :param i: index of first alpha
        :param error_i: e_i
        :return: index of alpha j
        """
        m = self.x.shape[0]
        max = 0
        error_i = self.get_error(i)
        for j in range(m):
            if j == i:
                continue
            error_j = self.get_error(j)
            if abs(error_j - error_i) > max:
                max = abs(error_j - error_i)
                alpha = j
        return alpha

    def edit_alpha(self, alpha1, alpha2, y1, y2, alpha_unc):
        """
        edit alpha
        :param alpha:
        :return:
        """
        L = 0
        H = 0
        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        elif y1 == y2:
            L = max(0, alpha2 + alpha1 - self.C)
            H = min(self.C, alpha2 + alpha1)
        if alpha_unc > H:
            return H
        elif (alpha_unc <= H) & (alpha_unc >= L):
            return alpha_unc
        else:
            return L

train = SMO(x, y, 0.6)
alpha_i = train.select_first_alpha()

# caculate kernel matrix
kernel_matrix = train.kernel_matrix

count = 0
iter_num = 50
iter = 0
while (alpha_i != None) & (iter < iter_num):
    iter += 1

    # find second alpha
    error_i = train.get_error(alpha_i)
    alpha_j = train.select_second_alpha(alpha_i, error_i)
    error_j = train.get_error(alpha_j)

    # update alpha
    eta = kernel_matrix[alpha_i, alpha_i] + kernel_matrix[alpha_j, alpha_j] - 2 * kernel_matrix[alpha_i, alpha_j]
    # eta = 0?
    alpha1_old = train.alphas[alpha_i, 0]
    alpha2_old = train.alphas[alpha_j, 0]
    alpha_unc = alpha2_old + y[alpha_j, 0] * (error_i - error_j) / eta
    print 'unc', alpha_unc
    train.alphas[alpha_j, 0] = train.edit_alpha(alpha1_old, alpha2_old, y[alpha_i, 0], y[alpha_j, 0], alpha_unc)

    train.alphas[alpha_i, 0] = alpha1_old + y[alpha_i, 0] * y[alpha_j, 0] * (alpha2_old - train.alphas[alpha_j, 0])

    # calculate b
    if 0 < train.alphas[alpha_i, 0] < train.C:
        b1 = -error_i - y[alpha_i, 0] * kernel_matrix[alpha_i, alpha_i] * (train.alphas[alpha_i, 0] - alpha1_old) -\
        y[alpha_j] * kernel_matrix[alpha_j, alpha_i] * (train.alphas[alpha_j, 0] - alpha2_old) + train.b

        b2 = -error_j - y[alpha_i, 0] * kernel_matrix[alpha_i, alpha_j] * (train.alphas[alpha_i, 0] - alpha1_old) - \
             y[alpha_j, 0] * kernel_matrix[alpha_j, alpha_j] * (train.alphas[alpha_j] - alpha2_old) + train.b

        train.b = (b1 + b2) / 2

print train.b


def plot_data(x, y):
    m = x.shape[0]
    ax = plt.axes(xlim=(-2, 12), ylim=(-8, 6))
    for i in range(m):
        if y[i, 0] == 1:
            plt.plot(x[i, 0], x[i, 1], 'bo')
        elif y[i, 0] == -1:
            plt.plot(x[i, 0], x[i, 1], 'rx')



fig = plt.figure()
plot_data(x, y)

m, n = x.shape

w = np.zeros((2, 1))
for i in range(m):
    w += np.multiply(train.alphas[i, 0] * y[i, 0], x[i, :].T)

x1 = -2
y1 = - (train.b + w[0, 0] * x1) / w[1, 0]
x2 = 12
y2 = - (train.b + w[0, 0] * x2) / w[1, 0]

y1 = y1[0, 0]
y2 = y2[0, 0]
plt.plot([x1, x2], [y1, y2])

plt.show()
