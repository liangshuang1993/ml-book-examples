from numpy import *

from _collections import defaultdict



class IIS(object):

    def __init__(self, threshold):

        self.w = [] # weight

        self.x = []

        self.y = []

        self.labels = [] # all the possibilities of y

        self.N = 0 # number of samples

        self.th = threshold

        self.features = defaultdict(int) # save all features f(x, y)=1, and number of examples in each feature

        self.n = 0 # number of features

        self.experience_e = []

        self.model_e = []

        self.M = 0



    def read_data(self, filename):

        f = open(filename, 'r')

        for line in f.readlines():

            sample = line.strip().split('\t')

            self.y.append(sample[0])

            self.x.append(sample[1:])

            # feature

            for feature in sample[1:]:

                self.features[(feature, sample[0])] += 1

        self.n = len(self.features.keys())

        self.N = len(self.y)

        self.labels = set(self.y)

        self.M = max(len(x) for x in self.x)



    def init_param(self):

        self.w = [0] * self.n

        self.experience_e = [0] * self.n

        self.update_experience_e()



    def get_z(self, x):

        z = 0

        for y in self.labels:

            tmp = 0

            for i in range(self.n):

                for single_x in x:

                    if (single_x, y) == self.features.keys()[i]:

                        # x, y satisify this feature

                        tmp += self.w[i]

            z += exp(tmp)

        #print 'z=', z

        return z



    def update_experience_e(self):

        for i in range(self.n):

            self.experience_e[i] = self.features.values()[i] / float(self.N)



    def update_model_e(self):

        self.model_e = [0] * self.n

        count = 0

        for x in self.x:

            for y in self.labels:

                p = self.get_conditional_prob(x, y)

                for single_x in x:

                    for i in range(self.n):

                        if (single_x, y) == self.features.keys()[i]:

                            result = p / float(self.N)

                            count += 1

                            #print '++++', i, self.model_e[i], result

                            self.model_e[i] += result

                            #print self.model_e[i]





    def get_conditional_prob(self, x, y):

        """

        get conditional probability

        :param x: features

        :param y: label

        :return: probability

        """

        z = self.get_z(x)

        tmp = 0



        for single_x in x:

            for i in range(self.n):

                if (single_x, y) == self.features.keys()[i]:

                    #print single_x, y

                    tmp += self.w[i]

        return exp(tmp) / z



    def update_w(self):

        self.update_model_e()

        for i in range(self.n):

            self.w[i] += log(self.experience_e[i] / self.model_e[i]) / float(self.M)



    def train_model(self, max_iter):

        self.init_param()

        for i in range(max_iter):

            last_w = self.w

            self.update_w()

            # check if converge

            for w, lw in zip(self.w, last_w):

                if fabs(w - lw) < self.th:

                    break



    def predict(self, data):

        x = data.split("\t")

        prob = {}

        for y in self.labels:

            prob[y] = self.get_conditional_prob(x, y)



        return prob





model = IIS(0.01)

model.read_data('data.txt')

model.train_model(100)

print model.predict("sunny\thot\thigh\tFALSE")


