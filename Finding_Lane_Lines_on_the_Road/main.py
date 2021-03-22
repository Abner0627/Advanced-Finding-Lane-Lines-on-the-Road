# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 16:38:45 2021

@author: Lab
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

#%% Func.
def perspective(img, src, dst):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size) 
    
    return warped

#%% Load pic 
P = './data'
F = 'test02.jpg'
img = cv2.imread(os.path.join(P, F))
# plt.imshow(img_gray, cmap="gray")

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

absX = cv2.convertScaleAbs(sobelx)
absY = cv2.convertScaleAbs(sobely)

dst_f = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
dst_f = cv2.cvtColor(dst_f, cv2.COLOR_BGR2GRAY)

# plt.imshow(dst_f, cmap="gray")

src = np.float32(
            [[280,  700],  # Bottom left
             [595,  460],  # Top left
             [725,  460],  # Top right
             [1125, 700]]) # Bottom right
        
dst = np.float32(
            [[250,  720],  # Bottom left
             [250,    0],  # Top left
             [1065,   0],  # Top right
             [1065, 720]]) # Bottom right   


offset =50


A = perspective(img)
plt.imshow(A, cmap="gray")

plt.show()