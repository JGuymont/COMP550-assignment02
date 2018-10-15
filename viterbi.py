import numpy as np

def add_one_smoothing(M):
    M += 1
    counts = np.sum(M, axis=1)
    for i in range(M.shape[0]):
        M[i] = M[i] / counts[i]
        if not sum(M[i]) - 1. < 1e-10:
            return 'Error, row {} do not sum to 1.'.format(i)
    return M

class Viterbi:

    def __init__(self, Pi, A, B, tags, lexicon):
        self.Pi = Pi
        self.A = A
        self.B = B
        self.tags = tags
        self.lexicon = lexicon
        self.n_tags = len(tags)
        self.lexicon_size = len(lexicon)

    def word_to_index(self, word):
        return self.lexicon.index(word)
    
    def get_trellis(self, sentence):
        words = sentence.split()
        n_words = len(words)

        trellis = np.zeros([self.n_tags, n_words])
        
        idx_word = self.word_to_index(words[0])
        for i in range(self.n_tags):
            trellis[i, 0] = self.Pi[i] * self.B[i, idx_word]

        for t in range(1, n_words):
            idx_word = self.word_to_index(words[t])
            for j in range(self.n_tags):
                trellis[j, t] = self.B[j, idx_word] * max([trellis[i, t-1] * self.A[i, j] for i in range(self.n_tags)])
        return trellis

    def predict_tags(self, sentence):
        trellis = self.get_trellis(sentence)
        tags_idx = np.argmax(trellis, axis=0)
        tags = [self.tags[i] for i in tags_idx]
        return tags