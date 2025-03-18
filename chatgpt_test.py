#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 00:00:09 2025

@author: cmalili
"""

import cv2
import numpy as np

# Load images
img1 = cv2.imread("rec1.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("rec1_transform1.png", cv2.IMREAD_GRAYSCALE)

# Step 1: Detect keypoints and compute descriptors
orb = cv2.ORB_create()  # Using ORB for feature detection
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Step 2: Match keypoints using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Step 3: Sort matches by distance (lower is better)
matches = sorted(matches, key=lambda x: x.distance)

# Step 4: Extract matched keypoints
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Step 5: Compute affine transformation using RANSAC
M, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC)

# Check if the matrix M is valid
if M is not None:
    M = M.astype(np.float32)  # Ensure it is of type float32

    # Apply affine transformation to img1
    h, w = img1.shape
    transformed_img1 = cv2.warpAffine(img1, M, (w, h))

    # Show matched keypoints (Optional)
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None)
    cv2.imshow("Matches", img_matches)
    cv2.imshow("Transformed Image", transformed_img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Affine transformation could not be computed.")