#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 18:23:50 2025

@author: cmalili
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

def recover_rigid_transform_linear(keypoints1, keypoints2, matches):
    """
    Recover rigid transformation (rotation and translation) from point correspondences
    using linear least squares.
    
    Parameters:
    -----------
    keypoints1, keypoints2 : ndarray
        Coordinates of keypoints from two images (y, x format)
    matches : list
        List of indices of matched keypoints [(idx1, idx2), ...]
        
    Returns:
    --------
    theta : float
        Estimated rotation angle in radians
    tx : float
        Estimated x-translation
    ty : float
        Estimated y-translation
    """
    # Extract matched points (converting from y,x to x,y format)
    points1 = []
    points2 = []
    for idx1, idx2 in matches:
        # Convert from y,x to x,y format
        y1, x1 = keypoints1[idx1]
        #y1, x1 = y1-150, x1-150
        y2, x2 = keypoints2[idx2]
        #y2, x2 = y2-150, x2-150
        points1.append([x1, y1])
        points2.append([x2, y2])
    
    points1 = np.array(points1)
    points2 = np.array(points2)
    
    # Construct matrices for linear least squares
    A = np.zeros((2 * len(matches), 3))
    b = np.zeros(2 * len(matches))
    
    for i in range(len(matches)):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        
        # Fill A matrix (for small angle approximation)
        A[2*i, 0] = -y1    # -y1 coefficient for theta in x equation
        A[2*i, 1] = 1      # 1 coefficient for tx in x equation
        A[2*i, 2] = 0      # 0 coefficient for ty in x equation
        
        A[2*i+1, 0] = x1   # x1 coefficient for theta in y equation
        A[2*i+1, 1] = 0    # 0 coefficient for tx in y equation
        A[2*i+1, 2] = 1    # 1 coefficient for ty in y equation
        
        # Fill b vector
        b[2*i] = x2 - x1   # x2 - x1 for x equation
        b[2*i+1] = y2 - y1 # y2 - y1 for y equation
    
    # Solve linear system using least squares
    params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # Extract parameters
    theta = params[0]  # in radians for small angle approximation
    tx = params[1]
    ty = params[2]
    
    return theta, tx, ty


# Demo function to show how to use the transformation recovery with the Harris feature detector
def demo_transform_recovery(image_path, transformed_image_path, k=0.04, threshold=0.3, nms_window=8):
    """
    Demonstrate the recovery of transformation parameters using Harris corner detection
    and feature matching.
    
    Parameters:
    -----------
    image_path : str
        Path to the original image
    transformed_image_path : str
        Path to the transformed image
    k : float
        Harris detector parameter
    threshold : float
        Threshold for corner detection
    nms_window : int
        Window size for non-maximum suppression
    """
    
    # Load images
    image1 = cv2.imread(image_path)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    
    image2 = cv2.imread(transformed_image_path)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    
    # Detect Harris corners
    corners1, response1 = harris_corner_detector(gray1, k=k, threshold=threshold)
    corners2, response2 = harris_corner_detector(gray2, k=k, threshold=threshold)
    
    # Apply non-maximum suppression
    corners1_suppressed = non_max_suppression(corners1, response1, window_size=nms_window)
    corners2_suppressed = non_max_suppression(corners2, response2, window_size=nms_window)
    
    # Compute descriptors
    descriptors1, keypoints1 = harris_keypoint_descriptor(gray1, corners1_suppressed)
    descriptors2, keypoints2 = harris_keypoint_descriptor(gray2, corners2_suppressed)
    
    # Match descriptors
    matches = match_descriptors(descriptors1, keypoints1, descriptors2, keypoints2, threshold=0.9)
    
    # Visualize matches
    match_image = visualize_matches(image1, keypoints1, image2, keypoints2, matches)
    
    # Recover transformation
    print(f"Number of matches: {len(matches)}")
    print("Using linear least squares:")
    if len(matches) >= 2:  # Need at least 2 correspondences
        theta_linear, tx_linear, ty_linear = recover_rigid_transform_linear(keypoints1, keypoints2, matches)
        print(f"Estimated rotation: {np.degrees(theta_linear):.2f} degrees")
        print(f"Estimated translation: ({tx_linear:.2f}, {ty_linear:.2f})")
        
        # Evaluate transformation
        #error_linear = evaluate_transformation(keypoints1, keypoints2, matches, theta_linear, tx_linear, ty_linear)
        #print(f"Average error (linear): {error_linear:.2f} pixels")
    
    
    # Display matches
    plt.figure(figsize=(15, 10))
    plt.imshow(match_image)
    plt.title(f'Feature Matching ({len(matches)} matches)')
    plt.axis('off')
    plt.show()
    

# Run the demo
if __name__ == "__main__":
    demo_transform_recovery("white_rec.jpg", "new_white_rec.jpg")