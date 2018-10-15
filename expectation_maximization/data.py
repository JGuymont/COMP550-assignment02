import numpy as np

tags = ['N', 'C', 'V', 'J']
lexicon = ['that', 'is', 'not', 'it', 'good', 'bad']

count_PI = np.array([0., 2., 2., 0.])

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