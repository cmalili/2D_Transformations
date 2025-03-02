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
def demo_harris_corner_detection(image_path=None, k=0.04, threshold=0.01, nms_window_size=8):
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
    if image_path:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # Create a test image with a simple shape
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        # Draw a white rectangle
        cv2.rectangle(image, (50, 50), (200, 200), (255, 255, 255), -1)
        # Draw a white triangle
        triangle_pts = np.array([[150, 30], [250, 120], [180, 250]], dtype=np.int32)
        cv2.fillPoly(image, [triangle_pts], (255, 255, 255))
    
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
    
    plt.subplot(222)
    plt.imshow(response, cmap='jet')
    plt.title('Harris Corner Response')
    plt.colorbar()
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
    

    # Also compare with OpenCV's implementation
    gray32 = np.float32(gray)
    harris_opencv = cv2.cornerHarris(gray32, blockSize=2, ksize=3, k=k)
    
    # Normalize and threshold the OpenCV result
    harris_opencv = cv2.dilate(harris_opencv, None)
    harris_opencv_norm = np.zeros_like(harris_opencv, dtype=np.float32)
    cv2.normalize(harris_opencv, harris_opencv_norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    opencv_corners = harris_opencv_norm > threshold
    
    # Visualize OpenCV implementation result
    image_opencv_corners = visualize_corners(image, opencv_corners, color=(255, 0, 0))
    
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(image_corners_suppressed)
    plt.title('Our Implementation')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(image_opencv_corners)
    plt.title('OpenCV Implementation')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    return corners_suppressed, response



def harris_keypoint_descriptor(image, keypoints, patch_size=9, num_bins=8, sigma=1.5):
    """
    Generate descriptors for Harris keypoints based on histogram of oriented gradients.
    
    Parameters:
    -----------
    image : ndarray
        Input grayscale image
    keypoints : ndarray
        Binary mask where True values indicate keypoint locations
    patch_size : int
        Size of the patch around each keypoint (must be odd)
    num_bins : int
        Number of orientation bins for the histogram
    sigma : float
        Standard deviation for Gaussian weighting
        
    Returns:
    --------
    descriptors : ndarray
        Array of descriptors for each keypoint
    keypoint_coords : ndarray
        Coordinates of keypoints that have valid descriptors
    """
    # Ensure patch_size is odd
    if patch_size % 2 == 0:
        patch_size += 1
    
    # Make sure image is grayscale
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Convert to float
    gray = gray.astype(np.float32)
    
    # Compute image gradients
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # Compute gradient magnitude and orientation
    magnitude = np.sqrt(dx**2 + dy**2)
    orientation = np.arctan2(dy, dx) * 180 / np.pi  # Convert to degrees
    
    # Adjust orientation to be in [0, 360) range
    orientation = np.mod(orientation, 360)
    
    # Get keypoint coordinates
    keypoint_coords = np.argwhere(keypoints)
    
    # Half patch size for neighborhood extraction
    half_patch = patch_size // 2
    
    # Initialize descriptors list
    descriptors = []
    valid_keypoints = []
    
    # Create a Gaussian weight matrix for the patch
    y, x = np.mgrid[-half_patch:half_patch+1, -half_patch:half_patch+1]
    gaussian_weights = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Process each keypoint
    for y, x in keypoint_coords:
        # Skip keypoints too close to the image boundary
        if (y < half_patch or y >= gray.shape[0] - half_patch or 
            x < half_patch or x >= gray.shape[1] - half_patch):
            continue
        
        # Extract patch around the keypoint
        y_start, y_end = y - half_patch, y + half_patch + 1
        x_start, x_end = x - half_patch, x + half_patch + 1
        
        patch_magnitude = magnitude[y_start:y_end, x_start:x_end]
        patch_orientation = orientation[y_start:y_end, x_start:x_end]
        
        # Weight magnitudes by Gaussian
        weighted_magnitude = patch_magnitude * gaussian_weights
        
        # Compute the histogram of oriented gradients
        hist_range = (0, 360)
        hist = np.zeros(num_bins)
        
        # Divide 360 degrees into num_bins
        bin_width = 360 / num_bins
        
        # Fill the histogram
        for i in range(patch_size):
            for j in range(patch_size):
                ori = patch_orientation[i, j]
                mag = weighted_magnitude[i, j]
                
                # Calculate bin index and contribution
                bin_idx = int(ori / bin_width) % num_bins
                hist[bin_idx] += mag
        
        # Normalize the histogram
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm
        
        # Add to descriptors
        descriptors.append(hist)
        valid_keypoints.append([y, x])
    
    return np.array(descriptors), np.array(valid_keypoints)

def match_descriptors(descriptors1, keypoints1, descriptors2, keypoints2, threshold=0.9):
    """
    Match descriptors between two sets of keypoints.
    
    Parameters:
    -----------
    descriptors1, descriptors2 : ndarray
        Descriptors from two images
    keypoints1, keypoints2 : ndarray
        Coordinates of keypoints from two images
    threshold : float
        Ratio test threshold for matching
        
    Returns:
    --------
    matches : list
        List of indices of matched keypoints [(idx1, idx2), ...]
    """
    matches = []
    
    # For each descriptor in the first image
    for i, desc1 in enumerate(descriptors1):
        # Compute distances to all descriptors in the second image
        distances = np.sqrt(np.sum((descriptors2 - desc1)**2, axis=1))
        
        # Sort distances
        sorted_indices = np.argsort(distances)
        
        # Apply ratio test (if first match is significantly better than second)
        if len(sorted_indices) >= 2:
            if distances[sorted_indices[0]] < threshold * distances[sorted_indices[1]]:
                matches.append((i, sorted_indices[0]))
    
    return matches

def visualize_matches(image1, keypoints1, image2, keypoints2, matches):
    """
    Visualize descriptor matches between two images.
    
    Parameters:
    -----------
    image1, image2 : ndarray
        Two images with keypoints
    keypoints1, keypoints2 : ndarray
        Coordinates of keypoints
    matches : list
        List of index pairs for matches
        
    Returns:
    --------
    match_image : ndarray
        Visualization of matches
    """
    # Create a combined image
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    combined_height = max(h1, h2)
    combined_width = w1 + w2
    match_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    
    # Place the images side by side
    match_image[:h1, :w1] = image1
    match_image[:h2, w1:w1+w2] = image2
    
    # Draw matches
    for idx1, idx2 in matches:
        y1, x1 = keypoints1[idx1]
        y2, x2 = keypoints2[idx2]
        
        # Draw points
        cv2.circle(match_image, (x1, y1), 4, (0, 255, 0), -1)
        cv2.circle(match_image, (x2 + w1, y2), 4, (0, 255, 0), -1)
        
        # Draw line
        cv2.line(match_image, (x1, y1), (x2 + w1, y2), (0, 255, 255), 1)
    
    return match_image

# Example of using the descriptor
def demo_harris_descriptor(image_path=None, transformed_image_path=None):
    """
    Demonstrate the Harris keypoint descriptor on an image and its transformed version.
    
    Parameters:
    -----------
    image_path : str
        Path to input image (if None, will create a test image)
    transformed_image_path : str
        Path to transformed version of the input image
    """
    # Create or load images
    if image_path:
        # Load the original image
        image1 = cv2.imread(image_path)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        
        # If no transformed image is provided, create a simple transformation
        if transformed_image_path:
            image2 = cv2.imread(transformed_image_path)
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        else:
            # Create a rotated and scaled version
            height, width = image1.shape[:2]
            center = (width//2, height//2)
            rotation_matrix = cv2.getRotationMatrix2D(center, 15, 0.9)
            image2 = cv2.warpAffine(image1, rotation_matrix, (width, height))
    else:
        # Create a test image with a simple shape
        image1 = np.zeros((300, 300, 3), dtype=np.uint8)
        # Draw a white rectangle
        cv2.rectangle(image1, (50, 50), (200, 200), (255, 255, 255), -1)
        # Draw a white triangle
        #triangle_pts = np.array([[150, 30], [250, 120], [180, 250]], dtype=np.int32)
        #cv2.fillPoly(image1, [triangle_pts], (255, 255, 255))
        
        # Create a rotated version
        height, width = image1.shape[:2]
        center = (width//2, height//2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 30, 0.9)
        image2 = cv2.warpAffine(image1, rotation_matrix, (width, height))
    
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    
    # Detect Harris corners in both images
    corners1, response1 = harris_corner_detector(gray1, k=0.04, threshold=0.3)
    corners2, response2 = harris_corner_detector(gray2, k=0.04, threshold=0.3)
    
    # Apply non-maximum suppression
    corners1_suppressed = non_max_suppression(corners1, response1, window_size=8)
    corners2_suppressed = non_max_suppression(corners2, response2, window_size=8)
    
    # Compute descriptors for keypoints
    descriptors1, keypoints1 = harris_keypoint_descriptor(gray1, corners1_suppressed)
    descriptors2, keypoints2 = harris_keypoint_descriptor(gray2, corners2_suppressed)
    
    # Match descriptors
    matches = match_descriptors(descriptors1, keypoints1, descriptors2, keypoints2)
    
    # Visualize matches
    match_image = visualize_matches(image1, keypoints1, image2, keypoints2, matches)
    
    # Display the visualization
    plt.figure(figsize=(15, 10))
    
    plt.subplot(221)
    plt.imshow(image1)
    plt.scatter(keypoints1[:, 1], keypoints1[:, 0], c='r', s=10)
    plt.title(f'Image 1 with {len(keypoints1)} Keypoints')
    plt.axis('off')
    
    plt.subplot(222)
    plt.imshow(image2)
    plt.scatter(keypoints2[:, 1], keypoints2[:, 0], c='r', s=10)
    plt.title(f'Image 2 with {len(keypoints2)} Keypoints')
    plt.axis('off')
    
    plt.subplot(212)
    plt.imshow(match_image)
    plt.title(f'Feature Matching ({len(matches)} matches)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    # For some descriptors, visualize their orientation histograms
    if len(descriptors1) > 0:
        plt.figure(figsize=(15, 5))
        
        # Display up to 5 descriptor histograms
        num_to_display = min(5, len(descriptors1))
        for i in range(num_to_display):
            plt.subplot(1, num_to_display, i+1)
            plt.bar(range(len(descriptors1[i])), descriptors1[i])
            plt.title(f'Descriptor {i+1}')
            plt.xticks(range(len(descriptors1[i])), 
                       [f"{j*360/len(descriptors1[i]):.0f}°" for j in range(len(descriptors1[i]))], 
                       rotation=45)
        
        plt.tight_layout()
        plt.show()



# Run the demo
if __name__ == "__main__":
    corners, response = demo_harris_corner_detection(k=0.04, threshold=0.3)
    demo_harris_descriptor()