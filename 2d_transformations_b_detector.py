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
    
    '''
    plt.subplot(222)
    plt.imshow(response, cmap='jet')
    plt.title('Harris Corner Response')
    plt.colorbar()
    plt.axis('off')
    '''
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



def harris_keypoint_descriptor(image, keypoints, patch_size=7, num_bins=8, sigma=1):
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
    
    gray = gray.astype(np.float32)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(dx**2 + dy**2)
    orientation = np.arctan2(dy, dx) * 180 / np.pi  # Convert to degrees
    
    orientation = np.mod(orientation, 360)
    
    keypoint_coords = np.argwhere(keypoints)
    
    half_patch = patch_size // 2
    
    descriptors = []
    valid_keypoints = []
    
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


def orientation_normalized_hkd(image, keypoints, patch_size=7, num_bins=8, sigma=1):
    if patch_size % 2 == 0:
        patch_size += 1  # Ensure odd patch size
    
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    gray = gray.astype(np.float32)
    
    # Compute gradients
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(dx**2 + dy**2)
    orientation = np.arctan2(dy, dx) * 180 / np.pi  # Convert to degrees
    orientation = np.mod(orientation, 360)  # Ensure values in [0, 360)
    
    keypoint_coords = np.argwhere(keypoints)
    half_patch = patch_size // 2
    descriptors = []
    valid_keypoints = []
    
    # Gaussian weighting
    y, x = np.mgrid[-half_patch:half_patch+1, -half_patch:half_patch+1]
    gaussian_weights = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    for y, x in keypoint_coords:
        if (y < half_patch or y >= gray.shape[0] - half_patch or 
            x < half_patch or x >= gray.shape[1] - half_patch):
            continue
        
        y_start, y_end = y - half_patch, y + half_patch + 1
        x_start, x_end = x - half_patch, x + half_patch + 1
        
        patch_magnitude = magnitude[y_start:y_end, x_start:x_end]
        patch_orientation = orientation[y_start:y_end, x_start:x_end]
        
        weighted_magnitude = patch_magnitude * gaussian_weights
        
        # Compute the histogram of oriented gradients
        num_bins = 8
        bin_width = 360 / num_bins
        #hist_range = (0, 360)
        hist = np.zeros(num_bins)
        
        for i in range(patch_size):
            for j in range(patch_size):
                ori = patch_orientation[i, j]
                mag = weighted_magnitude[i, j]
                
                bin_idx = int(ori / bin_width) % num_bins
                hist[bin_idx] += mag
        
        # Get the dominant orientation
        dominant_orientation = np.argmax(hist) * bin_width
        
        # Normalize patch orientation relative to dominant orientation
        patch_orientation = np.mod(patch_orientation - dominant_orientation, 360)
        
        # Compute histogram again with adjusted orientations
        hist = np.zeros(num_bins)
        for i in range(patch_size):
            for j in range(patch_size):
                ori = patch_orientation[i, j]
                mag = weighted_magnitude[i, j]
                
                bin_idx = int(ori / bin_width) % num_bins
                hist[bin_idx] += mag
        
        # Normalize the histogram
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm
        
        descriptors.append(hist)
        valid_keypoints.append([y, x])
    
    return np.array(descriptors), np.array(valid_keypoints)


def match_descriptors(descriptors1, keypoints1, descriptors2, keypoints2, threshold=0.3):
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
    # Load the original image
    image1 = cv2.imread(image_path)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    
    # If no transformed image is provided, create a simple transformation
    image2 = cv2.imread(transformed_image_path)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
   
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
    descriptors1, keypoints1 = orientation_normalized_hkd(gray1, corners1_suppressed)
    descriptors2, keypoints2 = orientation_normalized_hkd(gray2, corners2_suppressed)
    
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
    corners, response = demo_harris_corner_detection(image_path="new_white_rec.jpg", k=0.04, threshold=0.7)
    #demo_harris_descriptor(image_path="white_rec.jpg", transformed_image_path="new_white_rec.jpg")
    # Usage example (uncomment to use)
    #demo_transform_recovery("white_rec.jpg", "new_white_rec.jpg")