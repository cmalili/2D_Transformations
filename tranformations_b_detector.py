#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 15:56:28 2025

@author: cmalili
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def harris_corner_detector(image, k=0.04, threshold=0.01, window_size=3, sigma=1):
    """
    Implementation of the Harris Corner Detection algorithm.
    
    Parameters:
    -----------
    image : ndarray
        Input grayscale image
    k : float
        Harris detector free parameter, typically in range [0.04, 0.06]
    threshold : float
        Threshold for corner detection (relative to maximum corner response)
    window_size : int
        Size of the window for computing gradients
    sigma : float
        Standard deviation for Gaussian filter
        
    Returns:
    --------
    corners : ndarray
        Binary mask of detected corners
    response : ndarray
        Harris response (corner measure) for each pixel
    """
    # Check if image is grayscale, if not convert
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Convert to float for calculations
    gray = gray.astype(np.float32)
    
    # Compute image gradients using Sobel operator
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # Compute products of derivatives for the Harris matrix elements
    Ixx = dx * dx
    Ixy = dx * dy
    Iyy = dy * dy
    
    # Apply Gaussian filter to the Harris matrix elements
    Ixx = gaussian_filter(Ixx, sigma=sigma)
    Ixy = gaussian_filter(Ixy, sigma=sigma)
    Iyy = gaussian_filter(Iyy, sigma=sigma)
    
    # Calculate the determinant and trace of the Harris matrix
    det = Ixx * Iyy - Ixy * Ixy
    trace = Ixx + Iyy
    
    # Calculate the Harris response (corner measure)
    response = det - k * (trace ** 2)
    
    # Normalize response to [0, 1]
    response_min = np.min(response)
    response_max = np.max(response)
    if response_max > response_min:
        response = (response - response_min) / (response_max - response_min)
    
    # Apply threshold to find corners
    corners = response > threshold
    
    return corners, response

def non_max_suppression(corners, response, window_size=3):
    """
    Apply non-maximum suppression to corner points to get more isolated corner points.
    
    Parameters:
    -----------
    corners : ndarray
        Binary mask of detected corners
    response : ndarray
        Harris response at each pixel
    window_size : int
        Size of the suppression window
        
    Returns:
    --------
    corners_suppressed : ndarray
        Binary mask of corners after non-maximum suppression
    """
    # Get coordinates of all corners
    corner_coords = np.argwhere(corners)
    
    # Initialize suppressed corners
    corners_suppressed = np.zeros_like(corners)
    
    # Sort corners by response (highest first)
    corner_strengths = [response[y, x] for y, x in corner_coords]
    sorted_indices = np.argsort(corner_strengths)[::-1]
    
    # Apply non-maximum suppression
    for idx in sorted_indices:
        y, x = corner_coords[idx]
        
        # If this corner is still a candidate
        if corners[y, x]:
            corners_suppressed[y, x] = True
            
            # Suppress all corners in the neighborhood
            y_start = max(0, y - window_size // 2)
            y_end = min(corners.shape[0], y + window_size // 2 + 1)
            x_start = max(0, x - window_size // 2)
            x_end = min(corners.shape[1], x + window_size // 2 + 1)
            
            # Exclude the current corner
            neighborhood = corners[y_start:y_end, x_start:x_end].copy()
            neighborhood[y - y_start, x - x_start] = False
            
            # Suppress other corners in neighborhood
            corners[y_start:y_end, x_start:x_end] = corners[y_start:y_end, x_start:x_end] & ~neighborhood
    
    return corners_suppressed

def visualize_corners(image, corners, color=(0, 255, 0), radius=5, thickness=1):
    """
    Visualize detected corners on the original image.
    
    Parameters:
    -----------
    image : ndarray
        Original RGB image
    corners : ndarray
        Binary mask of detected corners
    color : tuple
        Color for corner markers (B, G, R)
    radius : int
        Radius of circle markers
    thickness : int
        Thickness of circle markers
        
    Returns:
    --------
    image_with_corners : ndarray
        Original image with corner markers
    """
    # Make a copy of the original image
    image_with_corners = image.copy()
    
    # If the image is grayscale, convert to RGB
    if len(image_with_corners.shape) == 2:
        image_with_corners = cv2.cvtColor(image_with_corners, cv2.COLOR_GRAY2BGR)
    
    # Draw circles at corner locations
    corner_coords = np.argwhere(corners)
    for y, x in corner_coords:
        cv2.circle(image_with_corners, (x, y), radius, color, thickness)
    
    return image_with_corners

# Example usage
def demo_harris_corner_detection(image_path='white_rec.jpg', k=0.04, threshold=0.01, nms_window_size=8):
    """
    Demonstrate the Harris Corner Detection algorithm on an image.
    
    Parameters:
    -----------
    image_path : str
        Path to input image (if None, will create a test image)
    k : float
        Harris detector parameter
    threshold : float
        Threshold for corner detection
    nms_window_size : int
        Window size for non-maximum suppression
    """
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
    
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(image_corners)
    plt.title('Detected Corners')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(image_corners_suppressed)
    plt.title('Corners After Non-Maximum Suppression')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    

    return corners_suppressed, response


# Run the demo
if __name__ == "__main__":
    corners, response = demo_harris_corner_detection(image_path="new_white_rec.jpg", k=0.04, threshold=0.7)
    