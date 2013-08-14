# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 10:59:35 2013

@author: attale00
"""

import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import FastICA,FactorAnalysis
from scipy import linalg
import featureGeneration as fg
import utils
 
 
## fetch natural image patches
#image_patches = fetch_mldata("natural scenes data")
#X = image_patches.data
# 
## 1000 patches a 32x32
## not so much data, reshape to 16000 patches a 8x8
#X = X.reshape(1000, 4, 8, 4, 8)
#X = np.rollaxis(X, 3, 2).reshape(-1, 8 * 8)

path = '/local/attale00/AFLW_ALL'
path_ea = path+'/color128/'
fileNames = utils.getAllFiles(path_ea);
path_mp = '/local/attale00/extracted_pascal__4__Multi-PIE/color128/'
mpFiles = utils.getAllFiles(path_mp)
    
    
roi=(50,74,96,160)
X=fg.getAllImagesFlat(path_ea,fileNames,(128,256),roi=roi)
#Y=fg.getAllImagesFlat(path_mp,mpFiles,(128,256),roi=roi)
#Z=np.concatenate((X,Y),axis=0)
Z=X
 ##perform ICA
ica = FastICA(n_components=49,whiten=True)
ica.fit(X)
filters = ica.unmixing_matrix_


#pca
## Center data
#n_samples, n_features = X.shape
#
#mean_ = np.mean(X, axis=0)
#X -= mean_
#U, S, V = linalg.svd(X)
#explained_variance_ = (S ** 2) / n_samples
#explained_variance_ratio_ = (explained_variance_ /explained_variance_.sum())
#K=V / S[:, np.newaxis] * np.sqrt(n_samples)
#filters=K[:49]

#factor analysis
#FA=FactorAnalysis(n_components=49)
#FA.fit(X)

 
# plot filters
plt.figure()
for i, f in enumerate(filters):
    plt.subplot(7, 7, i + 1)
    plt.imshow(f.reshape(roi[1]-roi[0], roi[3]-roi[2]), cmap="gray")
    plt.axis("off")
plt.show()