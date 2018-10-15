import numpy as np
from copy import deepcopy

from add_one_smoothing import add_one_smoothing
from data import tags, lexicon, count_PI, count_A, count_B 

class EM:

    def __init__(self, Pi, A, B, tags, lexicon, sentence):
        self.Pi = Pi
        self.A = A
        self.B = B
        self.tags = tags
        self.lexicon = lexicon
        self.lexicon_size = len(lexicon)
        self.words = sentence.split()
        self.N = len(tags)
        self.T = len(self.words)
        self.alpha = np.zeros([self.N, self.T])
        self.beta = np.ones([self.N, self.T])
        self.gamma = np.zeros([self.N, self.T])
        self.xi = np.zeros([self.T, self.N, self.N])

    def word_to_index(self, word):
        return self.lexicon.index(word)

    def forward(self):
        idx_word = self.word_to_index(self.words[0])
        for j in range(self.N):
            self.alpha[j, 0] = self.Pi[j] * self.B[j, 0]
        for t in range(1, self.T):
            idx_word = self.word_to_index(self.words[t])
            for j in range(self.N):
                self.alpha[j, t] = sum([self.alpha[i, t-1] * self.A[i,j] * self.B[j, idx_word] for i in range(self.N)])
        return self.alpha

    def backward(self):
        for t in range(self.T-2, -1, -1):
            idx_word = self.word_to_index(self.words[t+1])
            for i in range(self.N):
                self.beta[i, t] = sum([self.beta[j, t+1] * self.A[i,j] * self.B[j, idx_word] for j in range(self.N)])
        return self.beta

    def compute_likelihood(self):
        self.likelihood = sum(self.alpha[:,self.T-1])
        return self.likelihood
    
    def e_step_gamma(self):
        for i in range(self.N):
            for t in range(self.T):
                self.gamma[i, t] = self.alpha[i, t] * self.beta[i, t] / self.likelihood
        return self.gamma

    def compute_xi(self, t, i, j, idx_word):
        return self.alpha[i, t] * self.A[i,j] * self.B[j, idx_word] * self.beta[j, t+1] / self.likelihood
    
    def e_step_xi(self):
        for t in range(self.T-1):
            idx_word = self.word_to_index(self.words[t+1])
            for i in range(self.N):
                for j in range(self.N):
                    self.xi[t, i, j] = self.compute_xi(t, i, j, idx_word)
        return self.xi

    def update_pi(self):
        Pi = deepcopy(self.Pi)
        for i in range(self.N):
            Pi[i] = self.gamma[i, 0]
        return Pi
    
    def _xi_normalizer(self, i):
        normalizer = 0
        for t in range(self.T-1):
            for j in range(self.N):
                normalizer += self.xi[t, i, j]
        return normalizer

    def update_A(self):
        A = deepcopy(self.A)
        for i in range(self.N):
            for j in range(self.N):
                A[i,j] = sum(self.xi[0:(self.T-1), i, j]) / self._xi_normalizer(i)
        return A

    def update_B(self):
        B = deepcopy(self.B)
        for i in range(self.N):
            for j in range(self.lexicon_size):
                normalizer = sum(self.gamma[i])
                B[i, j] = 0
                for t in range(self.T):
                    idx_word = self.word_to_index(self.words[t])
                    B[i, j] = B[i, j] + self.gamma[i, t] if j == idx_word else B[i, j]
                B[i, j] = B[i, j] / normalizer
        return B

    def update_parameters(self):
        self.forward()
        self.backward()
        self.compute_likelihood()
        self.e_step_gamma()
        self.e_step_xi()
        Pi = self.update_pi()
        A = self.update_A()
        B = self.update_B()
        return Pi, A, B

if __name__ == '__main__':
    Pi = add_one_smoothing(count_PI)
    A =  add_one_smoothing(count_A)
    B =  add_one_smoothing(count_B)

    sentence1 = 'bad is not good'
    sentence2 = 'is it bad'
    train_data = [sentence1, sentence2]

    em = EM(Pi, A, B, tags, lexicon, sentence1)
    Pi1, A1, B1 = em.update_parameters()

    em = EM(Pi, A, B, tags, lexicon, sentence2)
    Pi2, A2, B2 = em.update_parameters()

    Pi = (Pi1 + Pi2) / 2
    A = (A1 + A2) / 2
    B = (B1 + B2) / 2
    
    print(sum(Pi))
    print(np.sum(A, axis=1))
    print(np.sum(B, axis=1))

