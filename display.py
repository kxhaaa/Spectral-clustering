# -*- coding: utf-8 -*-
"""
Created on Sat May 22 22:23:58 2021

@author: Xianghan Kong
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
    plt.title('Clustering result.')
    return np.uint8(result)

#usage
result1 = display_clusters(img, which_component)

























