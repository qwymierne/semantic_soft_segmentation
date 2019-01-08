import numpy as np
from imguidedfilter import imguidedfilter


def feature_PCA(features, dim):
    features = features.astype(float)
    h, w, d = features.shape
    features = features.reshape((-1, d))
    featmean = np.mean(features, axis=0)
    features = features - np.matmul(np.ones((h * w)), featmean)
    covar = np.matmul(features.T, features)
    eigen_values, eigen_vectors = np.linalg.eig(covar)
    idx = eigen_values.argsort()
    eigen_vectors = eigen_vectors[:, idx][:dim]
    pcafeat = np.matmul(features, eigen_vectors)
    pcafeat = pcafeat.reshape((h, w, dim))

    return pcafeat


def preprocess_features(features, img=None):
    features[features < -5] = -5
    features[features > 5] = 5

    if img is not None:
        fd = features.shape[3]
        maxfd = fd - fd % 3
        for i in range(0, maxfd, 3):
            #  features(:, :, i : i+2) = imguidedfilter(features(:, :, i : i+2), image, 'NeighborhoodSize', 10);
            features[:, :, i : i + 3] = imguidedfilter(features[:, :, i : i + 3], img, (10, 10), 0.01)
        for i in range(maxfd, fd):
            # features(:, :, i) = imguidedfilter(features(:, :, i), image, 'NeighborhoodSize', 10);
            features[:, :, i] = imguidedfilter(features[:, :, i], img, (10, 10), 0.01)

    simp = feature_PCA(features, 3)
    for i in range(0, 3):
        # simp(:,:,i) = simp(:,:,i) - min(min(simp(:,:,i)));
        simp[:, :, i] = simp[:, :, i] - simp[:, :, i].min()
        # simp(:,:,i) = simp(:,:,i) / max(max(simp(:,:,i)));
        simp[:, :, i] = simp[:, :, i] / simp[:, :, i].max()

    return simp
