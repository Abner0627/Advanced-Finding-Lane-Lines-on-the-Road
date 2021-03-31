import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

#%% Load video
P = './data'
V = 'tw_NH3.mp4'
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
             [430, 340],  # Top left
             [540, 340],  # Top right
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
             [595, 460],  # Top left
             [735, 460],  # Top right
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
             [740, 760],  # Top left
             [1123, 760],  # Top right
             [1730, 1080]]) # Bottom right

dst = np.float32(
            [[280, 1080],  # Bottom left
             [280, 0],  # Top left
             [1730, 0],  # Top right
             [1730, 1080]]) # Bottom right
''' 

#%% tw_NH3.mp4
'''
src = np.float32(
            [[300, 720],  # Bottom left
             [500, 540],  # Top left
             [680, 540],  # Top right
             [1010, 720]]) # Bottom right

dst = np.float32(
            [[300, 720],  # Bottom left
             [300, 0],  # Top left
             [1010, 0],  # Top right
             [1010, 720]]) # Bottom right
''' 