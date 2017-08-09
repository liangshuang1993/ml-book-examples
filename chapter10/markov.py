class Markov(object):
    def __init__(self):
        self.I = [] # state
        self.O = [] # observation
        self.Q = [1, 2, 3, 4]
        self.V = ['R', 'W']
        self.pai = [1.0 / 4] * 4
        self.A = [[0, 1, 0, 0], [0.4, 0, 0.6, 0], [0, 0.4, 0, 0.6], [0, 0, 0.5, 0.5]]
        self.B = [[0.5, 0.5], [0.3, 0.7], [0.6, 0.4], [0.8, 0.2]]

    def forward(self, t, i):
        pass

    def backward(self, t, i):
        pass
    