# -*- coding: utf-8 -*-
"""
Created on Fri May 28 20:29:59 2021

@author: 孔湘涵
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import function
import sklearn
import sklearn.metrics
# =============================================================================
# Data preprocessing
# =============================================================================
img =cv2.imread('024541_image.jpg')
#img = cv2.imread('092468_image.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(150,100))

img1 = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
plt.figure()
plt.imshow((img),cmap='gray')

row,col = img1.shape
data = img1.reshape((-1))

# =============================================================================
# Perform classic spectral clustering on grey scale and shrinking image
# =============================================================================
#U = function.spectral_cluster_slow(data)
U = np.load('trad_spec_results.npy')
which_component, centroids = function.k_means_1d(U[:,2].reshape(-1,1), function.k_means_pp_1d(U[:,2],k=8), n_iterations=200)

result_final = function.display_clusters(img, which_component)
result1_cv = cv2.cvtColor(result_final,cv2.COLOR_RGB2BGR)
#cv2.imwrite('classic_2.png',result1_cv)

print('Davies-Bouldin',sklearn.metrics.davies_bouldin_score(data.reshape(-1, 1), which_component))


















