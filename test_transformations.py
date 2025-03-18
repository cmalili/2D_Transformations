#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 20:07:32 2025

@author: cmalili
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


# 2D Transformation 1

affMatrix1 = np.array([[2, 0, -70]
                       [0, 2, -70]])

# 2D Transformation 1
center = (150, 150)  # Center of the image
rotation_matrix1 = cv2.getRotationMatrix2D(center, 30, 1.0)
affMatrix2 = 0.8*rotation_matrix1

# 2D Transformation 3
rotation_matrix2 = cv2.getRotationMatrix2D(center, 40, 1.0)
affMatrix2 = np.array([[1.5, 0, -80]
                       [0, 1.5, -20]])
affMatrix2 = affMatrix2 @ rotation_matrix2