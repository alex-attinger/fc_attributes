import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import FastICA, PCA
from sklearn.cluster import KMeans
from scipy import linalg
 
# fetch natural image patches
image_patches = fetch_mldata("natural scenes data")
X = image_patches.data
n, p = X.shape
# 1000 patches a 32x32
# not so much data, reshape to 16000 patches a 8x8
X = X.reshape(1000, 4, 8, 4, 8)
X = np.rollaxis(X, 3, 2).reshape(-1, 8 * 8)
 
# perform ICA
ica = FastICA(n_components=49)
ica.fit(X)
filters_ica = ica.unmixing_matrix_
 
# zero mean "by hand" so that inverse transform
# doesn't mess up the filters
#X -= X.mean(axis=0)
# perform whitening
n_samples, n_features = X.shape

# Center data
mean_ = np.mean(X, axis=0)
X -= mean_
U, S, V = linalg.svd(X)
explained_variance_ = (S ** 2) / n_samples
explained_variance_ratio_ = (explained_variance_ /explained_variance_.sum())
K=V / S[:, np.newaxis] * np.sqrt(n_samples)
K=K[:49]

#        if self.whiten:
#            self.components_ = V / S[:, np.newaxis] * np.sqrt(n_samples)
#        else:
#            self.components_ = V
##X1 = np.dot(K, X)
# see (13.6) p.267 Here X1 is white and data
# in X has been projected onto a subspace by PCA
#X_white =X1*np.sqrt(p)


#kmeans = KMeans(k=49, n_init=1).fit(X_white)
#pca.components_=K
#filters_kmeans = pca.inverse_transform(kmeans.cluster_centers_)
#filters_pca = pca.components_
 
titles = ["ICA", "PCA", "k-means"]
filters = [filters_ica,K]
 
for T, F in zip(titles, filters):
    plt.figure(T)
    for i, f in enumerate(F):
        plt.subplot(7, 7, i + 1)
        plt.imshow(f.reshape(8, 8), cmap="gray")
        plt.axis("off")
plt.show()