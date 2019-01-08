import numpy as np
from sklearn.cluster import KMeans
import math


def soft_segments_from_eigs(eig_vecs, laplacian, h, w, eig_vals=None, features=None, comp_cnt=10, max_iter=20,
                            sparsity_param=0.9, image_grad=None, initial_segments=None):

    eig_val_cnt = eig_vecs.shape[1]
    if eig_vals is None:
        eig_vals = 1e-5 * np.identity(eig_val_cnt)

    if initial_segments is None:
        if features is not None:
            features = features.reshape(-1, features.shape[2])  # TODO check
            initial_segments = KMeans(n_clusters=comp_cnt, max_iter=100, n_jobs=-1).fit_predict(features)
        else:
            eig_vals = abs(eig_vals)
            init_eigs_cnt = 20
            init_eigs_weights = np.diag(1 / np.sqrt(np.diag(eig_vecs[1:init_eigs_cnt + 1, 1:init_eigs_cnt + 1])))
            init_eig_vecs = eig_vecs[:, 1:init_eigs_cnt + 1] * init_eigs_weights  # TODO test
            initial_segments = KMeans(n_clusters=comp_cnt, max_iter=100, n_jobs=-1).fit_predict(features)

    soft_segments = np.zeros((len(initial_segments), comp_cnt))
    for i in range(comp_cnt):
        soft_segments[:, i] = (initial_segments == i).astype(float)

    print(soft_segments)
    print(soft_segments.shape)

    if image_grad is None:
        sp_mat = sparsity_param
    else:
        image_grad[image_grad > 0.2] = 0.2
        image_grad = image_grad + 0.8
        sp_mat = np.matlib.repmat(image_grad, comp_cnt, 1)

    thr_e = 1e-10
    w1 = 0.3
    w0 = 0.3
    e1 = w1 ** sparsity_param * np.power(np.maximum(abs(soft_segments), thr_e), (sp_mat - 2))
    print(e1)
    e0 = w0 ** sparsity_param * np.power(np.maximum(abs(soft_segments + 1), thr_e), (sp_mat - 2))

    scld = 1
    eig_vectors = eig_vecs[:, :eig_val_cnt]
    eig_values = eig_vals[:eig_val_cnt]

    remove_iter = math.ceil(max_iter / 4)
    remove_iter_cycle = math.ceil(max_iter / 4)

    for iter in range(max_iter):
        tA = np.zeros(((comp_cnt - 1) * eig_val_cnt,(comp_cnt - 1) * eig_val_cnt))
        tb = np.zeros(((comp_cnt - 1) * eig_val_cnt, 1))
        for k in range(comp_cnt - 1):
            weighted_eigs = np.matlib.repmat(e1[:, k] + e0[:, k], 1, eig_val_cnt) * eig_vectors  # czemu nie działa
            tA[k * eig_val_cnt:(k + 1) * eig_val_cnt, k * eig_val_cnt:(k + 1) * eig_val_cnt] = \
                np.matmul(eig_vectors.T, weighted_eigs) + scld * eig_values
            tb[k * eig_val_cnt:(k + 1) * eig_val_cnt] = np.matmul(eig_vectors.T, e1[:,k])
        k = comp_cnt
        weighted_eigs = np.matlib.repmat(e1[:, k] + e0[:, k], 1, eig_val_cnt) * eig_vectors
        ttA = np.matmul(eig_vectors.T, weighted_eigs + scld * eig_values)
        ttb = np.matmul(eig_vectors.T, e0[:, k]) + scld * np.sum(np.matmul(eig_vectors.T, laplacian), axis=1)

        tA = tA + np.matlib.repmat(ttA, comp_cnt - 1, comp_cnt - 1)
        tb = tb + np.matlib.repmat(ttb, comp_cnt - 1, 1)

        y = np.linalg.solve(tA, tb).reshape(eig_val_cnt, comp_cnt - 1)
        soft_segments = np.matmul(eig_vecs[:, :eig_val_cnt], y)
        soft_segments[:, comp_cnt - 1] = 1 - np.sum(soft_segments[:, :comp_cnt - 1], axis=1)

        if iter > remove_iter:
            nzii = filter((lambda x: x > 0.1), max(abs(soft_segments)) > 0.1)
            comp_cnt = len(nzii)
            soft_segments = soft_segments[:, nzii]
            remove_iter += remove_iter_cycle

        if len(sp_mat) == 1:
            e1 = w1 ** sparsity_param * np.power(np.maximum(abs(soft_segments), thr_e), (sp_mat - 2))
            e0 = w0 ** sparsity_param * np.power(np.maximum(abs(soft_segments + 1), thr_e), (sp_mat - 2))
        else:
            e1 = w1 ** sparsity_param * np.power(np.maximum(abs(soft_segments), thr_e), (sp_mat[:, :soft_segments.shape[1]] - 2))
            e0 = w0 ** sparsity_param * np.power(np.maximum(abs(soft_segments + 1), thr_e), (sp_mat[:, :soft_segments.shape[1]] - 2))
    soft_segments = soft_segments.reshape((h, w, soft_segments.shape[1]))
    return soft_segments
