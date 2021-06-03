# -*- coding: utf-8 -*-
"""
Created on Thu May 13 20:53:08 2021

@author: 孔湘涵
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import function
import time
import sklearn
import sklearn.metrics


# =============================================================================
# Data Preprocessing
# =============================================================================
img =cv2.imread('024541_image.jpg')

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
row,col = img.shape[:2]
channels = 3
plt.figure()
plt.imshow(img)

## lab space    
img_lab = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)   
data = img_lab.reshape((-1,channels))        

## RGB spce
#data = img.reshape((-1,channels))

# =============================================================================
# Spectral Clustering
# =============================================================================
time1 = time.time()

num_samples = 50
sample_indices, remain_indices = function.sample(row,col,num_samples)

#compute graph matrix, default method='fully'; method = -1, ε- neighborhood
A,B = function.similarity(data,sample_indices)

#Apply nystrom method
v = function.spectral_cluster_nystrom(A, B, sample_indices, remain_indices)
print('Total time is ', time.time()-time1)

# Cluster the points using k-means
k=8
which_component, centroids = function.k_means(v, function.k_means_pp(v,k), n_iterations=200)

#display the results
result1 = function.display_clusters(img, which_component)
result1_cv = cv2.cvtColor(result1,cv2.COLOR_RGB2BGR)
#cv2.imwrite('50_lab_fully_1.png',result1_cv)

# =============================================================================
# Evaluation of Performance
# =============================================================================
labelpath = '024541_inst_label.png'
label = cv2.imread(labelpath)[:,:,0].reshape(-1)
print(set(list(label)))
label[label==255]=1
result2 = function.display_clusters(img, label, k=7)

#labelpath = '092468_inst_label.png'
#label = cv2.imread(labelpath)[:,:,0].reshape(-1)
#print(set(list(label)))
#label[label==255]=7
#result2 = function.display_clusters(img, label, k=8)

print('Davies-Bouldin',sklearn.metrics.davies_bouldin_score(data, which_component))









