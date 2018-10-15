import numpy as np

def add_one_smoothing(M):
    M += 1
    counts = np.sum(M, axis=1) if len(M.shape) > 1 else np.sum(M)
    for i in range(M.shape[0]):
        M[i] = M[i] / counts[i] if len(M.shape) > 1 else M[i] / counts
    return M