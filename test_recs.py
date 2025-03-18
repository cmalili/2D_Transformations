#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 22:10:54 2025

@author: cmalili
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Quadrilateral number 1
# Create a 300x300 black image
img1 = np.zeros((300, 300, 3), dtype=np.uint8)
quad_points1 = np.array([
    [125, 130],   # Top-left
    [175, 125],   # Top-right
    [180, 170],   # Bottom-right
    [130, 180]    # Bottom-left
], dtype=np.int32)

cv2.fillPoly(img1, [quad_points1], (0, 0, 255))

center = (150, 150)  # Center of the image
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
img1 = cv2.warpAffine(img1, rotation_matrix, (300, 300))

# Quadrilateral number 2
quad_points2 = np.array([
    [150, 120],   # Top-left   # Top-right
    [180, 150],
    [150, 170],# Bottom-right
    [130, 140]    # Bottom-left
], dtype=np.int32)

img2 = np.zeros((300, 300, 3), dtype=np.uint8)
# Draw the quadrilateral (filled with white)
cv2.fillPoly(img2, [quad_points2], 255)

# Quadrilateral number 3
img3 = np.zeros((300, 300, 3), dtype=np.uint8)
quad_points3 = np.array([
    [125, 130],   # Top-left
    [175, 125],   # Top-right
    [180, 170],   # Bottom-right
    [130, 180]    # Bottom-left
], dtype=np.int32)

cv2.fillPoly(img3, [quad_points1], (0, 255, 0))

center = (150, 150)  # Center of the image
rotation_matrix = cv2.getRotationMatrix2D(center, 150, 1.0)
img3 = cv2.warpAffine(img3, rotation_matrix, (300, 300))

# Displaying the images of the rectangles
plt.figure(figsize=(12, 4))
plt.imshow(img1, cmap='gray')
plt.title('Quadrilateral 1')
plt.axis('on')
plt.show()

# Display the original image
plt.figure(figsize=(12, 4))
plt.imshow(img2, cmap='gray')
plt.title('Quadrilateral 2')
plt.axis('on')
plt.show()


plt.figure(figsize=(12, 4))
plt.imshow(img3, cmap='gray')
plt.title('Quadrilateral 3')
plt.axis('on')
plt.show()

cv2.imwrite("rec1.png", img1, [cv2.IMWRITE_PNG_COMPRESSION, 0])
cv2.imwrite("rec2.png", img2, [cv2.IMWRITE_PNG_COMPRESSION, 0])
cv2.imwrite("rec3.png", img3, [cv2.IMWRITE_PNG_COMPRESSION, 0])