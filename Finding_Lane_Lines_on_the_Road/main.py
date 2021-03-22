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

#%% Sobel filter
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

absX = cv2.convertScaleAbs(sobelx)
absY = cv2.convertScaleAbs(sobely)

dst_f = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
# plt.imshow(dst_f, cmap="gray")

#%% Transform
src = np.float32(
            [[235, 490],  # Bottom left
             [375, 375],  # Top left
             [570, 375],  # Top right
             [810, 490]]) # Bottom right

dst = np.float32(
            [[235, 490],  # Bottom left
             [235, 0],  # Top left
             [810, 0],  # Top right
             [810, 490]]) # Bottom right   

# dist1 = np.sqrt((375-235)^2 + (490-375)^2)
# print(dist1)
# dist2 = np.sqrt((810-570)^2 + (490-375)^2)
# print(dist2)

img_trans = perspective(dst_f, src, dst)

# fig, ax = plt.subplots(1,3)
# ax[0].imshow(img)
# ax[1].imshow(img_trans)
# ax[2].imshow(img)
# plt.tight_layout()

# plt.show()