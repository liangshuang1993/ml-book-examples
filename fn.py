from numpy import *

class NeuralNetwork(object):
    def __init__(self, input_num, hidden_num, output_num):
        self.input_num = input_num # input units number
        self.hidden_num = hidden_num #hidden units number, this network only one hidden layer
        self.output_num = output_num

        self.w1 = mat(zeros([input_num + 1, hidden_num])) # add one dimension
        self.w2 = mat(zeros([hidden_num + 1, output_num])) # add one dimension

        self.x = []
        self.y = []
        self.a1 = []
        self.a2 = []
        self.z0 = []
        self.z1 = []
        self.output = []

        self.m = 0

    def read_data(self, data_file):
        # y has only one dimension
        f = open(data_file, 'r')
        for line in f.readlines():
            self.x.append(line[:-1])
            self.y.append(line[-1])
            self.m += 1
        self.x = mat(self.x)
        self.y = mat(self.y)

    def init_param(self):
        # add one dimension
        self.z0 = hstack((self.x, ones([self.m, 1])))

    def forward(self, z0):
        self.a1 = z0 * self.w1
        self.z1 = self.sigmoid(self.a1)
        self.a2 = self.z1 * self.w2
        self.output = self.sigmoid(self.a2)

    def sigmoid(self, x):
        return 1.0 / (1 + exp(-x))

    def sigmoid_d(self, y):
        return y * (1 - y)

    def backpropagation(self):
        # update w2
        gradient2 = self.z1.T * multiply((self.output - self.y), self.sigmoid_d(self.output))
        #gradient1 =

    def train(self, lr, iter_num):
        self.init_param()
        for i in range(iter_num):
            self.forward(self.z0)
            gradient1, gradient2 = self.backpropagation()
            self.w1 += lr * gradient1
            self.w2 += lr * gradient2

    def predict(self, x):
        self.forward(x)



if __name__ == '__main__':
    model = NeuralNetwork(2, 3, 2)
    model.read_data('text.txt')
    model.train(0.01)



