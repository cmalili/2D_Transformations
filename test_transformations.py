#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 20:07:32 2025

@author: cmalili
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


# 2D Transformation 1

affMatrix1 = np.float32([[2, 0, -80],
                       [0, 2, -80]])

# 2D Transformation 2
center = (150, 150)  # Center of the image
rotation_matrix1 = cv2.getRotationMatrix2D(center, 30, 1.0)
affMatrix2 = 0.8*rotation_matrix1

# 2D Transformation 3
rotation_matrix2 = cv2.getRotationMatrix2D(center, 40, 1.0)
rotation_matrix2 = np.vstack([rotation_matrix2, [0, 0, 1]])
affMatrix3 = np.float32([[1.5, 0, -80],
                       [0, 1.5, -20],
                       [0, 0, 1]])
affMatrix3 = affMatrix3 @ rotation_matrix2
affMatrix3 = affMatrix3[:2]


# Load quadrilaterals
files = ["rec1.png", "rec2.png", "rec3.png"]
transformations = [affMatrix1, affMatrix2, affMatrix3]

for i, file in enumerate(files):
    rec = cv2.imread(file)
    rec = cv2.cvtColor(rec, cv2.COLOR_BGR2RGB)
    
    # Applying transformations to rectangles
    for j, transformation in enumerate(transformations):
        rec_transform = cv2.warpAffine(rec, transformation, (300, 300))
        
        plt.figure(figsize=(12, 4))
        plt.imshow(rec_transform, cmap='gray')
        plt.title(f'Quadrilateral {i + 1} under Transform {j + 1}')
        plt.axis('on')
        plt.show()
        
        name, ext = os.path.splitext(file)
        new_filename = f"{name}_transform{j+1}{ext}"
        cv2.imwrite(new_filename, rec_transform, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
