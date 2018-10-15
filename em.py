import numpy as np
from viterbi import Viterbi, add_one_smoothing

TAGS = ['N', 'C', 'V', 'J']
LEXICON = ['that', 'is', 'not', 'it', 'good', 'bad']

Pi = [1/8, 3/8, 3/8, 1/8]

count_A = np.array([
    [2., 0., 3., 1.],
    [2., 0., 0., 0.],
    [4., 0., 1., 0.],
    [0., 0., 0., 0.]
])

count_B = np.array([
    [4., 0., 2., 2., 0., 0.],
    [2., 0., 0., 0., 0., 0.],
    [0., 6., 0., 0., 0., 0.],
    [0., 0., 0., 0., 1., 0.]
])

if __name__ == '__main__':
    A = add_one_smoothing(count_A)
    B = add_one_smoothing(count_B)

    viterbi = Viterbi(Pi, A, B, TAGS, LEXICON)

    sentence1 = 'bad is not good'
    sentence2 = 'is it bad'
    pred1 = viterbi.predict_tags(sentence1)
    pred2 = viterbi.predict_tags(sentence2)
    print(pred1)
    print(pred2)


