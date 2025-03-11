#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:45:41 2025

@author: cmalili
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Create a 300x300 black image
img = np.zeros((300, 300), dtype=np.uint8)

# Define the four corners of an irregular quadrilateral
# Centered around (150, 150) with approximate size 50x50
quad_points = np.array([
    [125, 130],   # Top-left
    [175, 125],   # Top-right
    [180, 170],   # Bottom-right
    [130, 180]    # Bottom-left
], dtype=np.int32)

# Draw the quadrilateral (filled with white)
cv2.fillPoly(img, [quad_points], 255)
cv2.imwrite("white_rec.jpg", img)

# Display the original image
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title('Original Quadrilateral')
plt.axis('on')

# 1. Translate the image by (30, 100)
M_translate = np.float32([
    [1, 0, 30],
    [0, 1, 100]
])
translated_img = cv2.warpAffine(img, M_translate, (300, 300))

# Display the translated image
plt.subplot(132)
plt.imshow(translated_img, cmap='gray')
plt.title('Translated by (30, 100)')
plt.axis('on')

# 2. Rotate the translated image by 45 degrees around the center
center = (150, 150)  # Center of the image
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_img = cv2.warpAffine(translated_img, rotation_matrix, (300, 300))

cv2.imwrite("new_white_rec.jpg", rotated_img)

# Display the rotated image
plt.subplot(133)
plt.imshow(rotated_img, cmap='gray')
plt.title('Rotated by 45 degrees')
plt.axis('on')

plt.tight_layout()
plt.show()
