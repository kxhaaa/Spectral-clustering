# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 23:16:24 2021

@author: Wendell
"""
from collections import deque
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import argparse

# Neighbour (coordinates of) pixels, including the given pixel.
def get_neighbors(height, width, pixel):
   return np.mgrid[
      max(0, pixel[0] - 1):min(height, pixel[0] + 2),
      max(0, pixel[1] - 1):min(width, pixel[1] + 2)
      ].reshape(2, -1).T

def watershed(img):
   """
   This function will apply the watershed segmentation algorithm on the input image.
   :params: img: the image that need to segmented, we assume that it is a gray scale image.
   :return: labels: the labeled image indicating the label each pixel belongs to
   """
   # initialization
   MASK = -2 # threshold level
   WSHED = 0 # value of the pixels belonging to the watersheds
   INIT = -1 # initial value of the labeled image (output)
   INQUEUE = -3 # value assigned to the pixels when they are put into the queue
   img = np.array(img)
   height, width = img.shape[0], img.shape[1]
   total = height * width
   labels = np.full((height, width), INIT, np.int32)
   current_label = 0
   flag = False
   fifo = deque()
   # apply morphological reconstruction on the given image
   # define a kernel
   kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
   #dilation & erosion
   img_dil = cv2.dilate(img, kernel, iterations=3)
   img_ero = cv2.erode(img, kernel, iterations=3)
   img_grad = img_dil - img_ero
   reshaped_image = img_grad.reshape(total)
   # Coordinates of neighbour pixels for each pixel.
   pixels = np.mgrid[0:height, 0:width].reshape(2, -1).T
   neighbours = np.array([get_neighbors(height, width, p) for p in pixels])
   if len(neighbours.shape) == 3:
      # Case where all pixels have the same number of neighbours.
      neighbours = neighbours.reshape(height, width, -1, 2)
   else:
      # Case where pixels may have a different number of pixels.
      neighbours = neighbours.reshape(height, width) 
   # sort the pixels of ito1 in the increasing order of their gray values 
   # (in the range [hmjn, hmax]).
   indices = np.argsort(reshaped_image)
   sorted_image = reshaped_image[indices]
   sorted_pixels = pixels[indices]

   # self.levels evenly spaced steps from minimum to maximum.
   levels = np.linspace(sorted_image[0], sorted_image[-1], 256)
   level_indices = []
   current_level = 0

   # Get the indices that deleimit pixels with different values.
   for i in range(total):
      if sorted_image[i] > levels[current_level]:
         # Skip levels until the next highest one is reached.
         while sorted_image[i] > levels[current_level]: 
            current_level += 1
         level_indices.append(i)
   level_indices.append(total)

   start_index = 0
   for stop_index in level_indices:
      # Mask all pixels at the current level.
      for p in sorted_pixels[start_index:stop_index]:
         labels[p[0], p[1]] = MASK
         # Initialize queue with neighbours of existing basins at the current level.
         for q in neighbours[p[0], p[1]]:
            # p == q is ignored here because labels[p] < WSHD
            if labels[q[0], q[1]] >= WSHED:
               labels[p[0], p[1]] = INQUEUE
               fifo.append(p)
               break

      # Extend catchment basins.
      while fifo:
         p = fifo.popleft()
         # Label p by inspecting neighbours.
         for q in neighbours[p[0], p[1]]:
            # Don't set lab_p in the outer loop because it may change.
            lab_p = labels[p[0], p[1]]
            lab_q = labels[q[0], q[1]]
            if lab_q > 0:
               if lab_p == INQUEUE or (lab_p == WSHED and flag):
                  labels[p[0], p[1]] = lab_q
               elif lab_p > 0 and lab_p != lab_q:
                  labels[p[0], p[1]] = WSHED
                  flag = False
            elif lab_q == WSHED:
               if lab_p == INQUEUE:
                  labels[p[0], p[1]] = WSHED
                  flag = True
            elif lab_q == MASK:
               labels[q[0], q[1]] = INQUEUE
               fifo.append(q)

      # Detect and process new minima at the current level.
      for p in sorted_pixels[start_index:stop_index]:
         # p is inside a new minimum. Create a new label.
         if labels[p[0], p[1]] == MASK:
            current_label += 1
            fifo.append(p)
            labels[p[0], p[1]] = current_label
            while fifo:
               q = fifo.popleft()
               for r in neighbours[q[0], q[1]]:
                  if labels[r[0], r[1]] == MASK:
                     fifo.append(r)
                     labels[r[0], r[1]] = current_label

      start_index = stop_index

   return labels
   
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default="./watershed_results/024541_image.jpg")

    args = parser.parse_args()
    return args
   
if __name__ == "__main__":
    import sklearn.metrics
    args = parse_args()
    img = plt.imread(args.img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    start = time.time()
    labels = watershed(img)
    end = time.time()
    print('Running time:', end - start)
    img = cv2.imread(args.img_path)
    result1 = display_clusters(img, labels)
    print('Davies-Bouldin:',sklearn.metrics.davies_bouldin_score(img.reshape((-1,3)), labels.reshape((-1,))))
    
    
    
    
    
    
    