# -*- coding: utf-8 -*-
"""
Created on Tue Jun 1 2021

@author: Weishen Li
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from glob import glob
from display import display_clusters
import sklearn.metrics

def k_means_pp(X, k):
    '''
    Compute initial custer for k-means
    arguments:
     - X:          np.ndarray of shape [no_data, no_dimensions]
                   input data points
    returns:
     - centroids:  np.ndarray of shape [k, no_dimensions]
                   centres of initial custers
    '''
    n_data = X.shape[0]
    k = min(n_data, k)
    centroids = []
    rand_i = np.random.choice(n_data)

    for _ in range(k):
        centroids.append(X[rand_i])
        X = np.concatenate((X[:rand_i], X[rand_i+1:]))

        distances = np.linalg.norm(np.expand_dims(X, axis=1) - centroids, axis=-1, ord=2).min(axis=1)
        rand_i = np.random.choice(X.shape[0], p=distances**2/(distances**2).sum())

    return np.array(centroids)

def kernel_kmeans(X, centroids, n_iterations):
    '''
    kernel k-means algorithm
    arguments:
     - X:          np.ndarray of shape [no_data, no_dimensions]
                   input data points
     - centroids:  np.ndarray of shape [k, no_dimensions]
                   centres of initial custers
     - n_iterations: integer, number of iterations to run k-means for
    returns:
     - which_component: np.ndarray of shape [no_data] and integer data
                        type, contains values in [0, k-1] indicating which
                        cluster each data point belongs to
     - centroids:  np.ndarray of shape [k, no_dimensions] centres of
                   final custers, ordered in such way as indexed by
                   `which_component`
    '''
    # using standard k-means to assign each point to a cluster centroid
    k = centroids.shape[0]
    nn = X.shape[0]
    distances = np.linalg.norm(np.expand_dims(X, axis=1) - centroids, axis=-1, ord=2)
    which_component = np.argmin(distances, axis=-1)

    # calculate the kernel matrix of data X
    kernel_matrix = np.zeros((nn,nn))
    sigma = np.var(img_array)
    for i in range(nn):
        kernel_matrix[i] = np.exp((-np.linalg.norm((img_array[i,:] - img_array), axis = 1)**2)/sigma)
    
    # calcuate centroid for each component
    for iter in range(n_iterations):
        #print("############### Iteration: %d ###############" % iter)
        dist = np.zeros((nn,k))
        # traverse each point in a cluster and then calculate the distances
        for kk in range(k):
            #print("Cluster: %d" % kk)
            # find out each point in cluster kk 
            masks = (which_component==kk)
            nnum = np.where(masks.T.squeeze() == True)
            #print(nnum[0][0])
            cluster_X = X[nnum[0]]
            k_matrix = kernel_matrix[nnum[0]]
            num_c = cluster_X.shape[0]
            # calculate the second term and the third term of the expanding formula in report
            for i in range(num_c):
                term2 = 2 * k_matrix[i,:].sum(axis = 0) / num_c
                term3 = k_matrix.sum().sum() / 2 / (num_c**2)
                # notice that the first term phi(x_i)*phi(x_i) is a constant 1
                dist[nnum[0][i],kk] = 1 - term2 + term3
        #re-assign the each point's cluster centroid then update the centroids
        which_component = np.argmin(dist, axis=-1)
        centroids = np.stack((X[which_component == 0].mean(axis=0),
                              X[which_component == 1].mean(axis=0),
                              X[which_component == 2].mean(axis=0),
                              X[which_component == 3].mean(axis=0)), axis=0)
        
    return which_component, centroids


"""
Main program starts from here
"""
# set random seed for PRNG
np.random.seed(int(time.time()))
# set up the image paths
image_paths = glob('images_for_test/*_image.jpg')
print(image_paths[0][16:])
num = len(image_paths)
# read in all the image files in the testing directory 
for n in range(num):
    # record the starting running time
    start = time.time()
    # read in the image
    image_frame = cv2.cvtColor(cv2.imread(image_paths[n],cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    image_frame2 = cv2.resize(image_frame, (150, 106))
    img = cv2.cvtColor(image_frame2, cv2.COLOR_RGB2LAB)
    height, width, channel = img.shape
    # compress the image then use an array to save it
    img_array = np.zeros((width * height, channel))
    for i in range(height):
        img_array[i * width:(i + 1) * width] = img[i]
    # set up the cluster numbers 
    k = 4
    nn = img_array.shape[0]
    # call the kernel_kmeans algorithm
    which_component, centers = kernel_kmeans(img_array, k_means_pp(img_array, k), n_iterations=10)
    # record the ending running time and print out the total using time
    end = time.time()
    print("Runtime: %f" %(end - start))
    # call sklearn's davies_bouldin_score and print out the results
    print('Davies-Bouldin',sklearn.metrics.davies_bouldin_score(img_array, which_component))
    # call an imported function to plot the 
    result = display_clusters(image_frame2, which_component,k)

    plt.show()
    cv2.imwrite('kernel_k-means_results/kkmeans_' + image_paths[0][16:], result)
