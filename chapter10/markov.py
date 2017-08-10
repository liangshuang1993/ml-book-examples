from numpy import mat, multiply, row_stack

class Markov(object):
    def __init__(self):
        self.I = [] # state
        self.O = mat([1, 0, 1]) # observation
        self.Q = mat([1, 2, 3])
        self.V = mat([1, 0]) # red white
        self.pai = mat([0.2, 0.4, 0.4])
        self.A = mat([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
        self.B = mat([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])

        self.T = self.O.shape[1]

    def forward(self, t, i):
        pass

    def backward(self, t, i):
        pass

    def trans_o(self):
        """
        change self.O to index of V
        :return:
        """
        temp_array = []
        for j in range(self.O.shape[1]):
            for i in range(self.V.shape[1]):
                if self.V[0, i] == self.O[0, j]:
                    temp_array.append(i)
        self.O = mat(temp_array)

    def predict(self):
        m, n = self.B.shape # m is number of state, n is number of obervation

        self.trans_o()

        # t = 1

        delta_1 = multiply(self.pai, self.B[:, self.O[0, 0]].T)
        delta = delta_1


        # initialize fai
        tmp = [0] * m
        fai = [tmp]
        # t from 1 to self.T
        for t in range(1, self.T):
            delta_t = []
            fai_t = []
            for i in range(m): # for each state Q[i]
                max = 0
                state = None
                for j in range(m):
                    if delta[t-1, j] * self.A[j, i] > max:
                        max = delta[t-1, j] * self.A[j, i]
                        state = j
                delta_t.append(max * self.B[i, self.O[0, t]])
                fai_t.append(state)
            fai.append(fai_t)
            delta = row_stack((delta, delta_t))

       # print fai

        # choose state
        max = 0
        states = []

        # find final state
        opt_state = []
        for i in range(m):
            if delta[self.T - 1, i] > max:
                max = delta[self.T - 1, i]
                opt_state = i
        states.append(opt_state)

        # from final state to first state
        for t in range(self.T - 2, -1, -1):
            opt_state = fai[t + 1][states[self.T - 2 - t]]
            states.append(opt_state)

        states.reverse()
        print states




if __name__ == "__main__":
    my_model = Markov()
    my_model.predict()
