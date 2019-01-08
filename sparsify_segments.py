import numpy as np


def sparsify_segments(soft_segments, laplacian, image_grad=None):

    sigmaS = 1  # sparsity
    sigmaF = 1  # fidelity
    delta = 100  # constraint
    h, w, comp_cnt = soft_segments.shape
    N = h * w * comp_cnt

    if image_grad is None:
        sp_pow = 0.9
    else:
        image_grad[image_grad > 0.1] = 0.1
        image_grad = image_grad + 0.9
        sp_pow = np.matlib.repmat(image_grad, comp_cnt, 1)
