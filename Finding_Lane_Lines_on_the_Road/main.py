# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 16:38:45 2021

@author: Lab
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

#%% Load pic 
P = './data'
F = 'test02.jpg'
img = cv2.imread(os.path.join(P, F))
# plt.imshow(img_gray, cmap="gray")

# convert to hls
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gaus = cv2.GaussianBlur(img_gray,(5,5),0)

# Canny on straight_1
# Use the lightness layer to detect lines
low_thresh = 50
high_thresh = 150


minLineLength = 100
maxLineGap = 10

# Lightness thresholding (returns less points than saturation thresholding, gives better representation of lane lines)
edges_lightness = cv2.Canny(img_gaus, high_thresh, low_thresh)
lines = cv2.HoughLinesP(edges_lightness, 1, np.pi/180, 150, None, minLineLength, maxLineGap)

Lhs = np.zeros((2, 2), dtype = np.float32)
Rhs = np.zeros((2, 1), dtype = np.float32)
x_max = 0
x_min = 2555
for line in lines:
    for x1, y1, x2, y2 in line:
        # Find the norm (the distances between the two points)
        normal = np.array([[-(y2-y1)], [x2-x1]], dtype = np.float32) # question about this implementation
        normal = normal / np.linalg.norm(normal)
        
        pt = np.array([[x1], [y1]], dtype = np.float32)
        
        outer = np.matmul(normal, normal.T)
        
        Lhs += outer
        Rhs += np.matmul(outer, pt) #use matmul for matrix multiply and not dot product

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness = 1)
        
        x_iter_max = max(x1, x2)
        x_iter_min = min(x1, x2)
        x_max = max(x_max, x_iter_max)
        x_min = min(x_min, x_iter_min)

width = x_max - x_min
print('width : ', width)
# Calculate Vanishing Point
vp = np.matmul(np.linalg.inv(Lhs), Rhs)

print('vp is : ', vp)
plt.plot(vp[0], vp[1], 'c^')
plt.imshow(img)
plt.title('Vanishing Point visualization')


plt.show()