from preprocess_features import preprocess_features
from spectral_matting import matting_affinity
from affinity_matrix_to_laplacian import affinity_matrix_to_laplacian
from soft_segments_from_eigs import soft_segments_from_eigs

import scipy
from scipy.misc import imread
import numpy as np


def semantic_soft_segmentation(img, features):
    # img = imread(image_path, mode='RGB')
    if features.shape[2] > 3:
        features = preprocess_features(features, img)
    # else:
    #     features = imread(features)

    # superpixels = Superpixels(img)
    h, w, _ = img.shape
    affinities = list()
    affinities.append(matting_affinity(img))
    # affinities[1] = superpixels.neighbor_affinities(features)
    # affinities[2] = superpixels.nearby_affinities(img)
    laplacian = affinity_matrix_to_laplacian(affinities[0]
                                             # + 0.01 * affinities[1]
                                             # + 0.01 * affinities[2]
                                             )

    eig_cnt = 100
    eigen_values, eigen_vectors = scipy.sparse.linalg.eigs(laplacian, k=eig_cnt, which='LM')
    # idx = eigen_values.argsort()
    # eigen_values = eigen_values[idx][:eig_cnt]
    # eigen_vectors = eigen_vectors[:, idx][:eig_cnt]

    initial_segm_cnt = 40
    sparsity_param = 0.8
    iter_cnt = 40
    init_soft_segments = soft_segments_from_eigs(eigen_vectors, laplacian, h, w, eigen_values, features,
                                                 initial_segm_cnt, iter_cnt, sparsity_param)
    # grouped_segments = group_segments(init_soft_segments, features)
    # soft_segments = sparsify_segments(grouped_segments, laplacian, image_gradient(img, False, 6))

    return (
        # soft_segments,
        init_soft_segments,
        laplacian,
        affinities,
        features,
        # superpixels,
        eigen_vectors,
        eigen_values
    )
