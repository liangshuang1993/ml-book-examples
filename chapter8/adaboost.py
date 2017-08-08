from numpy import mat, log, exp


class Single_decision_tree(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.min = mat(x).min()
        self.max = mat(x).max()
        self.th = None
        self.mode = None

    def gate(self, x, th, mode):
        if mode == 'lt':
            if x >= th:
                return 1
            else:
                return -1
        elif mode == 'st':
            if x <= th:
                return 1
            else:
                return -1

    def train(self, weights):
        min_loss = None
        for _ in ('lt', 'st'):
            for th in range(self.min, self.max):
                loss = 0
                for i, x in enumerate(self.x):
                    if (self.gate(x, th, _) * self.y[i] == -1):
                        loss += weights[i]
                if min_loss == None:
                    min_loss = loss
                    self.th = th
                    self.mode = _
                elif loss < min_loss:
                    min_loss = loss
                    self.th = th
                    self.mode = _


    def predict(self, x):
        if self.mode == 'lt':
            if x >= self.th:
                return 1
            else:
                return -1
        elif self.mode == 'st':
            if x <= self.th:
                return 1
            else:
                return -1


class Adaboost(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.g = [] # predict
        self.alpha = [] # G(x) weights
        self.n = len(self.x)
        self.th = []
        self.model = []

    def weak_learning(self, weights):
        # single layer decision tree
        model = Single_decision_tree(self.x, self.y)
        self.model.append(model)
        model.train(weights)
        self.th.append(model.th)
        g = []
        for x in self.x:
            g.append(model.predict(x))
        return g

    def update_error(self, g, weights):
        error = 0
        for i in range(self.n):
            if g[i] != self.y[i]:
                error += weights[i]
        return error

    def update_alpha(self, error):
        alpha = 0.5 * log((1 - error) / error)
        self.alpha.append(alpha)
        return alpha

    def update_weights(self, old_weights, alpha, g):
        z = 0
        weights = []
        for i in range(self.n):
            tmp = old_weights[i] * exp(-alpha * self.y[i] * g[i])
            weights.append(tmp)
            z += tmp
        for i in range(len(weights)):
            weights[i] = weights[i] / z
        return weights


    def train(self, iternum):
        # init weights
        weights = [1.0 / self.n] * self.n

        for m in range(iternum):
            g = self.weak_learning(weights)

            # update error
            error = self.update_error(g, weights)

            # update alpha
            alpha = self.update_alpha(error)

            # update weights
            weights = self.update_weights(weights, alpha, g)

    def predict(self, x):
        m = len(self.alpha)
        f = 0
        for i in range(m):
            # using weak classifier
            g = self.model[i].predict(x)
            f += self.alpha[i] * g
        if f >= 1:
            return 1
        else:
            return -1

f = open('data.txt', 'r')
x = []
y = []
for line in f.readlines():
    tmp = line.split()
    x.append(int(tmp[0]))
    y.append(int(tmp[1]))
adaboost_model = Adaboost(x, y)
adaboost_model.train(4)
print adaboost_model.predict(1)

