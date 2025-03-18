#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 21:13:14 2025

@author: cmalili
"""

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

files = ["rec1.png", "rec2.png", "rec3.png"]
files_transformation1 = ['rec1_transform1.png', 'rec2_transform1.png', 'rec3_transform1.png']
files_transformation2 = ['rec1_transform2.png', 'rec2_transform2.png', 'rec3_transform2.png']
files_transformation3 = ['rec1_transform3.png', 'rec2_transform3.png', 'rec3_transform3.png']


# Load images of the original and transformed rectangles
img1 = cv2.imread("rec3.png", cv2.IMREAD_GRAYSCALE)  # Original rectangle
img2 = cv2.imread("rec1_transform1.png", cv2.IMREAD_GRAYSCALE)  # Transformed rectangle

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

if des1 is None or des2 is None:
    print("Error: No descriptors found. Check if keypoints were detected.")
    sys.exit()
    
des1 = np.float32(des1)
des2 = np.float32(des2)


# Use FLANN based matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Match descriptors
matches = flann.knnMatch(des1, des2, k=2)

# Apply Lowe’s ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Ensure enough matches exist
if len(good_matches) > 4:
    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute the transformation (Affine or Homography)
    M, mask = cv2.estimateAffine2D(src_pts, dst_pts)  # Use this for affine transformation
    # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)  # Use this for perspective transform

    print("Estimated Affine Transformation Matrix:\n", M)

    # Draw matches for visualization
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(12, 4))
    plt.imshow(img_matches, cmap='gray')
    plt.title('Quadrilateral 3')
    plt.axis('on')
    plt.show()
else:
    print("Not enough matches found!")
