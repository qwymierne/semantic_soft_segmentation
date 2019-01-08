import numpy as np
import numpy.matlib
import skimage
import scipy.ndimage
from scipy.misc import imread
import matplotlib.pyplot as plt


def local_RGB_normal_distributions(img, window_radius=1, epsilon=1e-8):
    h, w, c = img.shape
    N = h * w
    window_size = 2 * window_radius + 1

    img_mean = scipy.ndimage.generic_filter(img, np.mean, size=(window_size, window_size, 1))
    cov_mat = np.zeros((3, 3, N))
    # cov_mat = (np.transpose(img - img_mean, (1, 2, 0)).reshape(-1, 2)).apply(np.cov)
    for r in range(c):
        for s in range(r, c):
            temp = scipy.ndimage.generic_filter(img[:, :, r] * img[:, :, s], np.mean, size=(window_size, window_size)) \
                   - img_mean[:, :, r] * img_mean[:, :, s]
            cov_mat[r, s] = temp.reshape((-1))

    for i in range(c):
        cov_mat[i, i] += epsilon

    for r in range(1, c):
        for s in range(r):
            cov_mat[r, s] = cov_mat[s, r]

    return img_mean, cov_mat


def matting_affinity(img, in_map=None, window_radius=1, epsilon=1e-7):
    window_size = 2 * window_radius + 1
    neigh_size = window_size ** 2
    h, w, c = img.shape
    if in_map is None:
        in_map = np.full((h, w), True)
    N = h * w
    epsilon = epsilon / neigh_size

    img_mean, cov_mat = local_RGB_normal_distributions(img, window_radius, epsilon)

    indices = np.array(range(h * w)).reshape(h, w)
    neigh_ind = skimage.util.view_as_windows(indices, window_size)
    in_map = in_map[window_radius:-window_radius, window_radius:-window_radius]
    neigh_ind = neigh_ind[in_map].reshape(-1, neigh_size)
    in_ind = neigh_ind[:, neigh_size // 2]
    pix_cnt = in_ind.shape[0]

    # Prepare in & out data
    img = img.transpose((2, 0, 1)).reshape(-1, 3)
    img_mean = img_mean.transpose((2, 0, 1)).reshape(-1, 3)
    flow_rows = np.zeros((neigh_size, neigh_size, pix_cnt))
    flow_cols = np.zeros((neigh_size, neigh_size, pix_cnt))
    flows = np.zeros((neigh_size, neigh_size, pix_cnt))

    # Compute matting affinity
    for i in range(pix_cnt):
        neighs = neigh_ind[i]
        shifted_win_colors = img[neighs] - np.matlib.repmat(img_mean[in_ind[i]], neighs.shape[0], 1)
        flows[:, :, i] = np.matmul(shifted_win_colors, np.linalg.solve(cov_mat[:, :, in_ind[i]], shifted_win_colors.T))
        neighs = np.matlib.repmat(neighs, neighs.shape[0], 1)
        flow_rows[:, :, i] = neighs
        flow_cols[:, :, i] = neighs.T

    flows = (flows + 1) / neigh_size

    W = scipy.sparse.coo_matrix((flows.reshape(-1), (flow_rows.reshape(-1), flow_cols.reshape(-1))), shape=[N, N])
    W = (W + W.T) / 2

    print(W.shape)
    return W


if __name__ == '__main__':
    matting_affinity('COCO_train2014_000000362884.jpg')
