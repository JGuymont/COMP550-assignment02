TAGS = ['N', 'C', 'V', 'J']
LEXICON = ['that', 'is', 'not', 'it', 'good']

Pi = [1/8, 3/8, 3/8, 1/8]

count_A = np.array([
    [2., 0., 3., 1.],
    [2., 0., 0., 0.],
    [4., 0., 1., 0.],
    [0., 0., 0., 0.]
])

count_B = np.array([
    [4., 0., 2., 2., 0.],
    [2., 0., 0., 0., 0.],
    [0., 6., 0., 0., 0.],
    [0., 0., 0., 0., 1.]
])

if __name__ == '__main__':
    A = add_one_smoothing(count_A)
    B = add_one_smoothing(count_B)

    viterbi = Viterbi(Pi, A, B, TAGS, LEXICON)

    sentence = 'that is good'

    pred = viterbi.predict_tags(sentence)
    print(pred)