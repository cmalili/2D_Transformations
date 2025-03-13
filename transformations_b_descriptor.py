#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 17:04:27 2025

@author: cmalili
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


from transformations_b_detector import harris_corner_detector
from transformations_b_detector import non_max_suppression
#from transformations_b_detector import orientation_normalized_hkd


def difference_of_gaussians(image, sigma1=1.0, sigma2=2.0):
    """Compute Difference of Gaussians (DoG)"""
    blur1 = gaussian_filter(image, sigma=sigma1)
    blur2 = gaussian_filter(image, sigma=sigma2)
    return blur2 - blur1

def compute_orientation_dog(image, sigma1=1.0, sigma2=2.0):
    """Compute gradient-based orientation using Difference of Gaussians (DoG)"""
    
    # Convert to grayscale if needed
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    gray = gray.astype(np.float32)

    # Compute DoG
    dog = difference_of_gaussians(gray, sigma1, sigma2)

    # Compute gradients
    Gx = cv2.Sobel(dog, cv2.CV_32F, 1, 0, ksize=3)  # X-gradient
    Gy = cv2.Sobel(dog, cv2.CV_32F, 0, 1, ksize=3)  # Y-gradient

    # Compute gradient magnitude and orientation
    magnitude = np.sqrt(Gx**2 + Gy**2)
    orientation = np.arctan2(Gy, Gx) * 180 / np.pi  # Convert to degrees
    orientation = np.mod(orientation, 360)  # Ensure values are in [0, 360]

    return magnitude, orientation



def harris_keypoint_descriptor(image, keypoints, patch_size=9, num_bins=9, sigma=1):
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
    '''
    #gray = gray - cv2.GaussianBlur(gray, (5,5), 2)
    
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(dx**2 + dy**2)
    orientation = np.arctan2(dy, dx) * 180 / np.pi  # Convert to degrees
    
    orientation = np.mod(orientation, 360)
    '''
    
    magnitude, orientation = compute_orientation_dog(gray)
    
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



def orientation_normalized_hkd(image, keypoints, patch_size=8, num_bins=30, sigma=1):
    if patch_size % 2 == 0:
        patch_size += 1  # Ensure odd patch size
    
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    gray = gray.astype(np.float32)
    magnitude, orientation = compute_orientation_dog(gray)
    
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
        #num_bins = 8
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


def match_descriptors(descriptors1, keypoints1, descriptors2, keypoints2, threshold=0.99):
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
    #descriptors1, keypoints1 = orientation_normalized_hkd(gray1, corners1_suppressed)
    #descriptors2, keypoints2 = orientation_normalized_hkd(gray2, corners2_suppressed)
    
    
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
        
    # For some descriptors, visualize their orientation histograms
    if len(descriptors2) > 0:
        plt.figure(figsize=(15, 5))
        
        # Display up to 5 descriptor histograms
        num_to_display = min(5, len(descriptors2))
        for i in range(num_to_display):
            plt.subplot(1, num_to_display, i+1)
            plt.bar(range(len(descriptors2[i])), descriptors2[i])
            plt.title(f'Descriptor {i+1}')
            plt.xticks(range(len(descriptors2[i])), 
                       [f"{j*360/len(descriptors1[i]):.0f}°" for j in range(len(descriptors2[i]))], 
                       rotation=45)
        
        plt.tight_layout()
        plt.show()
        
    for i, match in enumerate(matches):
        print(f"match: {match[0],match[1]}")



# Run the demo
if __name__ == "__main__":
    demo_harris_descriptor(image_path="white_rec.png", 
                           transformed_image_path="new_white_rec.png")
