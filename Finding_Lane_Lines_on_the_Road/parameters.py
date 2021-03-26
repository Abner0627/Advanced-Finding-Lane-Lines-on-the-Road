import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

#%% Load video
P = './data'
V = 'solidWhiteRight.mp4'
cap = cv2.VideoCapture(os.path.join(P, V))
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    vd = frame
    break

fig_0 = plt.figure(0)
plt.imshow(vd)
plt.show()

#%% solidWhiteRight.mp4
'''
src = np.float32(
            [[153, 540],  # Bottom left
             [430, 337],  # Top left
             [536, 337],  # Top right
             [872, 540]]) # Bottom right

dst = np.float32(
            [[153, 540],  # Bottom left
             [153, 0],  # Top left
             [872, 0],  # Top right
             [872, 540]]) # Bottom right 
'''

#%% solidYellowLeft.mp4
'''
src = np.float32(
            [[100, 540],  # Bottom left
             [280, 430],  # Top left
             [675, 430],  # Top right
             [840, 540]]) # Bottom right

dst = np.float32(
            [[100, 540],  # Bottom left
             [100, 0],  # Top left
             [840, 0],  # Top right
             [840, 540]]) # Bottom right
'''

#%% challenge.mp4
'''
src = np.float32(
            [[280, 666],  # Bottom left
             [503, 517],  # Top left
             [828, 517],  # Top right
             [1080, 666]]) # Bottom right

dst = np.float32(
            [[280, 666],  # Bottom left
             [280, 0],  # Top left
             [1080, 0],  # Top right
             [1080, 666]]) # Bottom right
'''

#%% tw_NH1.mp4
'''
src = np.float32(
            [[280, 1080],  # Bottom left
             [709, 780],  # Top left
             [1162, 780],  # Top right
             [1730, 1080]]) # Bottom right

dst = np.float32(
            [[280, 1080],  # Bottom left
             [280, 0],  # Top left
             [1730, 0],  # Top right
             [1730, 1080]]) # Bottom right
''' 