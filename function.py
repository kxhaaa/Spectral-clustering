# -*- coding: utf-8 -*-
"""
Created on Wed May  5 17:00:59 2021

@author: 孔湘涵
"""
import numpy as np
import time
import random
import matplotlib.pyplot as plt

# =============================================================================
# Spectral clustering/ Normalized cut functions
# =============================================================================

def spectral_cluster_slow(data):
    '''
    The classic spetral clustering algorithm.
    arguments:
     - data:       np.ndarray of shape [no_data, no_dimensions]
                   input data points
    returns:
     - U:          top k eigenvectors, np.ndarray of shape [no_data, k]  
    '''
    time1 = time.time()
    number = data.shape[0]
    #Step 1: compute matrix W
    w1 = np.broadcast_to(data, (number,number)) #w1[:,0,:] is same
    w2 = w1.T
    w = np.float32(w1)-np.float32(w2)
    W = np.exp(-w**2)
    time2 = time.time()
    print('Time of compute matrix W is ',time2-time1) 
               
    #Step2: compute matrix D
    d = W.sum(axis = 0)
    D = np.diag(d)
    time3 = time.time()
    print('Time of compute matrix D is ',time3-time2) 
    
    #Step3: compute Graph Laplacian matrix
    L = np.linalg.inv(D)**0.5 @ (D - W) @ np.linalg.inv(D)**0.5
    time4 = time.time()
    print('Time of compute matrix L is ',time4-time3)
    
    #Step4: do eigenvalue decomposition of L
    values, vectors = np.linalg.eigh(L)
    time5 = time.time()
    print('Time of compute eigendecomposition is ',time5-time4)

    #Step5: find k smallest eigenvalues
    gap = np.zeros(len(values)-1)
    for i in range(gap.shape[0]):
        gap[i] = values[i+1] - values[i]
    k = np.argmax(gap)+1
    U = vectors[:,:k]
    print('Total time is ',time.time()-time1)
 
    return U

def spectral_cluster_nystrom(A, B, sample_indices, remain_indices):
    '''
    The fast spetral clustering algorithm using Nystrom method.
    arguments:
     - A:               similarity sub-matrix shape [no_samples, no_samples]
     - B:               similarity sub-matrix shape [no_samples, no_remaining_points]
     -sample_indices:   np.ndarray of shape [no_samples]
     -remain_indices:   np.ndarray of shape [no_remaining_points]
                   
    returns:
     - V:               top k eigenvectors, np.ndarray of shape [no_data, k] 
    '''
    num_points = A.shape[1] + B.shape[1]
    num_samples = sample_indices.shape[0]
    
    #1. compute row sums of w which is d, and reset the samples location at front
    d1 = np.sum(A,axis=1) + np.sum(B,axis=1)
    d2 = np.sum(B,axis=0) + np.dot(B.T,np.dot(np.linalg.pinv(A),np.sum(B,axis=1)))
    dhat = np.reshape(np.sqrt(1/np.concatenate([d1,d2])),[num_points,1])
    
    #2. get new A & B
    A = A * np.dot(dhat[0:num_samples],dhat[0:num_samples].T)
    B = B * np.dot(dhat[0:num_samples],dhat[num_samples:].T)
    
    #3.compute s and diagonalize it
    Asi = np.linalg.pinv(A**0.5)
    BBT  = np.dot(B,B.T)
    S = A + np.dot(Asi,np.dot(BBT,Asi))
    us,gammas,_ = np.linalg.svd(S)
    gammas = np.diag(gammas)

    #4, choose the first k singular vectors
    k = 8
    ABT = np.zeros((num_points,num_samples))
    ABT[sample_indices,:] = A
    ABT[remain_indices,:] = B.T
    V = ABT @ Asi @ us[:,1:k] @ np.linalg.pinv(gammas[1:k,1:k]**0.5)   
    v = V / np.broadcast_to(np.linalg.norm(V,axis=1).reshape(-1,1), (V.shape))    #data normalization
    return v

# =============================================================================
# Similarity matrix construction functions
# =============================================================================

def sample(row,col,num_sample):
    '''
    Produce some samples.
    arguments:
     - row,col:         image size.
     - num_sample:      no_samples, number of samples
                   
    returns:
     - sample_indices:  np.ndarray of shape [no_samples]
     - remain_indices:  np.ndarray of shape [no_remaining_points]
    '''
    sample_indices = np.array(random.sample(range(row*col), num_sample))
    remain_indices = np.delete(range(row*col), sample_indices)
    return sample_indices, remain_indices

def similarity(data, sample_indices, method='fully'):
    '''
    Compute similarity sub-matrix A & B.
    arguments:
     - data:            np.ndarray of shape [no_data, no_dimensions]
     - sample_indices   np.ndarray of shape [no_samples]
     - method           choose the type of similaritymatrix, default is 'fully connected graph',
                        if want to use 'ε- neighborhood graph', set method as other value
                   
    returns:
     - A                similarity sub-matrix shape [no_samples, no_samples]
     - B                similarity sub-matrix shape [no_samples, no_remaining_points]
    
    Hint: data needs normalized to 1, in case AB=0.
    '''
    data=np.float32(data/data.max())
    length = data.shape[0]
    AB = np.zeros((len(sample_indices),length)) 
    samples = data[sample_indices,:]
    sigma=1   
    for i in range(len(sample_indices)):
        # use Gaussian kernel to define the similarity
        AB[i,:] = np.exp((-np.linalg.norm((samples[i,:] - data), axis = 1)**2)/sigma)    #fully connected  sigma=1 
    if method != 'fully':
        AB = AB[AB>np.exp(-0.8)**2].reshape((len(sample_indices),length))    #ε- neighborhood default is 0.8
        print('ε- neighborhood graph.')
    else:
        print('fully connected graph.')
    A = AB[:,sample_indices]
    B = AB[:,np.delete(range(length), sample_indices)]
    return A,B

# =============================================================================
# K-means functions
# =============================================================================

def k_means_1d(X, centroids, n_iterations):
    '''
    standard k-means algorithm
    arguments:
     - X:          np.ndarray of shape [no_data]
                   input data points
     - centroids:  np.ndarray of shape [k]
                   centres of initial custers
     - n_iterations: integer, number of iterations to run k-means for
    returns:
     - which_component: np.ndarray of shape [no_data] and integer data
                        type, contains values in [0, k-1] indicating which
                        cluster each data point belongs to
     - centroids:  np.ndarray of shape [k], centres of 
                   final custers, ordered in such way as indexed by
                   `which_component`
    '''
    k = centroids.shape[0]
    for _ in range(n_iterations):
        # reassign data points to components
        distances = np.linalg.norm(np.expand_dims(X, axis=1) - centroids, axis=-1, ord=2)
        
        which_component = np.argmin(distances, axis=-1)
        # calcuate centroid for each component
        centroids = np.stack(list( X[which_component==i].mean(axis=0) for i in range(k) ), axis=0)

    return which_component, centroids

def k_means_pp_1d(X, k):
    '''
    Compute initial custer for k-means
    arguments:
     - X:          np.ndarray of shape [no_data]
                   input data points
    returns:
     - centroids:  np.ndarray of shape [k]
                   centres of initial custers
    '''
    channels = 1
    num_data = X.shape[0]    

    #step1: get a random point as the first center
    index1 = int(np.random.random_sample()*num_data) 
    centroids = np.zeros((k,channels))
    centroids[0] = X[index1]    
    index = np.zeros(k)  #the index of centers in dataset
    index[0] = index1
    for i in range(1,k):
        #step2: compute every point's distance to the nearest existing centroid
        distance = np.ones(num_data)          
        for j in range(num_data):   #for all data
            dis = np.ones(i+1)
            for m in range(0,i+1):
                #each distance between center m and every point
                dis[m] = np.linalg.norm(X[j] - centroids[m])
            #assign each point to the nearest center with minimum distance
            distance[j] = dis.min()    
            if distance[j] == 0:
                distance[j] += 1e-5
        #step3: choose one point as the centre of a new cluster with probability proportional to distance**2
        index[i] = np.argmax(distance) 
        centroids[i] = X[int(index[i])]
    return centroids

def k_means(X, centroids, n_iterations):
    '''
    standard k-means algorithm
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
     - centroids:  np.ndarray of shape [k, no_dimensions], centres of 
                   final custers, ordered in such way as indexed by
                   `which_component`
    '''
    k = centroids.shape[0]
    for _ in range(n_iterations):
        # reassign data points to components
        distances = np.linalg.norm(np.expand_dims(X, axis=1) - centroids, axis=-1, ord=2)
        which_component = np.argmin(distances, axis=-1)
        # calcuate centroid for each component
        centroids = np.stack(list( X[which_component==i].mean(axis=0) for i in range(k) ), axis=0)

    return which_component, centroids
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
    num_data, channels = X.shape    

    #step1: get a random point as the first center
    index1 = int(np.random.random_sample()*num_data) 
    centroids = np.zeros((k,channels))
    centroids[0,:] = X[index1,:]    
    index = np.zeros(k)  #the index of centers in dataset
    index[0] = index1
    for i in range(1,k):
        #step2: compute every point's distance to the nearest existing centroid
        distance = np.ones(num_data)          
        for j in range(num_data):   #for all data
            dis = np.ones(i+1)
            for m in range(0,i+1):
                #each distance between center m and every point
                dis[m] = np.linalg.norm(X[j,:] - centroids[m,:])
            #assign each point to the nearest center with minimum distance
            distance[j] = dis.min()    
            if distance[j] == 0:
                distance[j] += 1e-5
        #step3: choose one point as the centre of a new cluster with probability proportional to distance**2
        index[i] = np.argmax(distance) 
        centroids[i,:] = X[int(index[i]),:]
    return centroids

# =============================================================================
# Display functions
# =============================================================================

def display_clusters(img, which_component, k=-1):
    '''
    Display the cluster result as the row image color.
    
    Param:
    img                  color RGB image, row*col*channels
    which_component      1d size = row*col, each point represent which cluster belongs to.
    (k)                  set by default, number of clusters
    '''
    row,col = img.shape[:2]
    which_component = which_component.astype(np.int64)
    if k==-1:
        k=which_component.max()+1
    else:
        pass
    center_value = np.zeros((k,3))
    result = np.zeros(img.shape)
    #calculate the mean value of each clusters
    for n in range(k):
        mask = np.array([which_component==n]).reshape((row,col))
        number = mask.sum()
        center_value[n,0] = (mask*img[:,:,0]).sum()/number
        center_value[n,1] = (mask*img[:,:,1]).sum()/number
        center_value[n,2] = (mask*img[:,:,2]).sum()/number
        result[:,:,0] += mask*center_value[n,0]
        result[:,:,1] += mask*center_value[n,1]
        result[:,:,2] += mask*center_value[n,2]
    plt.figure()
    plt.imshow(np.uint8(result))   
#    plt.title('Clustering result (RGB & fully).')
    return np.uint8(result)




















