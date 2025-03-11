#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 18:28:04 2025

@author: cmalili
"""

from feature_detector_and_descriptor import *

image_path = "white_rec.jpg"
k=0.04
threshold=0.3
nms_window_size=8

 # Load or create test image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 # Convert to grayscale for Harris detector
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
 # Detect corners using Harris algorithm
corners, response = harris_corner_detector(gray, k=k, threshold=threshold)
 # Apply non-maximum suppression
corners_suppressed = non_max_suppression(corners, response, window_size=nms_window_size)
 # Visualize results
image_corners = visualize_corners(image, corners, color=(0, 255, 0))
image_corners_suppressed = visualize_corners(image, corners_suppressed, color=(0, 0, 255))
 
 # Show corner response heatmap
plt.figure(figsize=(16, 10))
 
plt.subplot(221)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
 
plt.subplot(223)
plt.imshow(image_corners)
plt.title('Detected Corners')
plt.axis('off')
 
plt.subplot(224)
plt.imshow(image_corners_suppressed)
plt.title('Corners After Non-Maximum Suppression')
plt.axis('off')
 
plt.tight_layout()
plt.show()
 

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
 
plt.subplot(132)
plt.imshow(image_corners_suppressed)
plt.title('Our Implementation')
plt.axis('off')
 
plt.tight_layout()
plt.show()