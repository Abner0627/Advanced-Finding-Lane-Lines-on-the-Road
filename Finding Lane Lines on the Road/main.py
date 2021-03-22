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
img_gray = cv2.imread(os.path.join(P, F), cv2.IMREAD_GRAYSCALE)

