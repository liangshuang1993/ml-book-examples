"""
P157
"""

class EM(object):
    def __init__(self, y, th):
        self.y = y # y is observation variable
        self.pai = 0
        self.p = 0
        self.q = 0
        self.th = th
        self.m = len(y)

    def init_param(self, pai, p, q):
        self.pai = pai
        self.p = p
        self.q = q

    def train(self):
        delta = 1000
        while delta > self.th:
            u = []
            # save old param
            old_pai = self.pai
            old_p = self.p
            old_q = self.q

            # estimate
            for i in range(self.m):
                tmp1 = self.pai * (self.p ** self.y[i]) * ((1 - self.p) ** (1 - self.y[i]))
                tmp2 = (1 - self.pai) * (self.q ** self.y[i]) * ((1 - self.q) ** (1 - self.y[i]))
                u.append(tmp1 / float(tmp1 + tmp2))


            # update model
            self.pai = sum(u)/float(self.m)

            from numpy import multiply, array
            self.p = sum(multiply(u, self.y)) / float(sum(u))
            self.q = sum(multiply(array([1] * self.m) - array(u), self.y)) / float(sum(array([1] * self.m) - array(u)))

            # caculate delta
            delta = (old_pai - self.pai) ** 2 + (old_p - self.p) ** 2 + (old_q - self.q) ** 2


f = open('data.txt', 'r')
y = []
for line in f.readlines():
    y.append(int(line.strip()))
model = EM(y, 0.0001)

# set init param = 0.5, 0.5, 0.5
model.init_param(0.5, 0.5, 0.5)
model.train()
print model.pai, model.p, model.q

# set init param = 0.4, 0.6, 0.7
model.init_param(0.4, 0.6, 0.7)
model.train()
print model.pai, model.p, model.q
print 'end'

