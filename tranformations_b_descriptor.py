#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 17:04:27 2025

@author: cmalili
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_harris_keypoints(image, threshold=0.01, neighborhood_size=3):
    """
    Detect Harris corners and return keypoints.
    
    Args:
        image: Grayscale input image
        threshold: Harris corner response threshold
        neighborhood_size: Size of the local non-maximum suppression window
        
    Returns:
        List of keypoint coordinates (y, x)
    """
    # Compute Harris corner response
    harris_response = cv2.cornerHarris(image.astype(np.float32), 
                                       blockSize=2, 
                                       ksize=3, 
                                       k=0.04)
    
    # Normalize the response
    harris_response = cv2.normalize(harris_response, None, 0, 1, cv2.NORM_MINMAX)
    
    # Apply threshold and non-maximum suppression
    keypoints = []
    for y in range(neighborhood_size, harris_response.shape[0] - neighborhood_size):
        for x in range(neighborhood_size, harris_response.shape[1] - neighborhood_size):
            if harris_response[y, x] > threshold:
                # Check if it's a local maximum
                window = harris_response[y-neighborhood_size:y+neighborhood_size+1, 
                                         x-neighborhood_size:x+neighborhood_size+1]
                if harris_response[y, x] == np.max(window):
                    keypoints.append((y, x))
    
    return keypoints

def compute_rotation_invariant_descriptor(image, keypoint, num_bins=36, radius_bins=4, max_radius=12):
    """
    Compute a rotation-invariant descriptor for a Harris corner keypoint.
    Uses a histogram of gradients in polar coordinates.
    
    Args:
        image: Grayscale input image
        keypoint: (y, x) coordinates of the keypoint
        num_bins: Number of orientation bins (angular resolution)
        radius_bins: Number of radial bins
        max_radius: Maximum radius to consider around the keypoint
        
    Returns:
        Descriptor vector (normalized histogram)
    """
    y, x = keypoint
    height, width = image.shape
    
    # Calculate image gradients
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude and orientation
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * 180 / np.pi  # Convert to degrees
    
    # Initialize the descriptor histogram
    histogram = np.zeros((radius_bins, num_bins))
    
    # Compute bin size for radius
    radius_bin_size = max_radius / radius_bins
    
    # Define a dominant orientation for rotation normalization
    dominant_orientation = 0
    orientation_votes = np.zeros(num_bins)
    
    # Collect votes for dominant orientation from the local region
    for i in range(-max_radius, max_radius + 1):
        for j in range(-max_radius, max_radius + 1):
            yi, xi = y + i, x + j
            
            # Check if the pixel is within image boundaries
            if 0 <= yi < height and 0 <= xi < width:
                pixel_radius = np.sqrt(i**2 + j**2)
                
                if pixel_radius <= max_radius:
                    # Calculate the orientation bin
                    ori = orientation[yi, xi]
                    # Ensure orientation is between 0 and 360
                    if ori < 0:
                        ori += 360
                    ori_bin = int(ori * num_bins / 360) % num_bins
                    
                    # Weight by magnitude
                    orientation_votes[ori_bin] += magnitude[yi, xi]
    
    # Find the dominant orientation (smoothed)
    smoothed_votes = np.copy(orientation_votes)
    for i in range(num_bins):
        # Circular smoothing with neighbors
        prev_idx = (i - 1) % num_bins
        next_idx = (i + 1) % num_bins
        smoothed_votes[i] = 0.25 * orientation_votes[prev_idx] + \
                             0.5 * orientation_votes[i] + \
                             0.25 * orientation_votes[next_idx]
    
    dominant_bin = np.argmax(smoothed_votes)
    dominant_orientation = dominant_bin * 360 / num_bins
    
    # Build the rotation-normalized descriptor
    for i in range(-max_radius, max_radius + 1):
        for j in range(-max_radius, max_radius + 1):
            yi, xi = y + i, x + j
            
            # Check if the pixel is within image boundaries
            if 0 <= yi < height and 0 <= xi < width:
                pixel_radius = np.sqrt(i**2 + j**2)
                
                if pixel_radius <= max_radius:
                    # Calculate the radius bin
                    r_bin = min(int(pixel_radius / radius_bin_size), radius_bins - 1)
                    
                    # Calculate the orientation relative to dominant orientation
                    ori = orientation[yi, xi]
                    if ori < 0:
                        ori += 360
                    
                    # Normalize the orientation with respect to the dominant orientation
                    normalized_ori = (ori - dominant_orientation) % 360
                    ori_bin = int(normalized_ori * num_bins / 360) % num_bins
                    
                    # Add weighted contribution to the histogram
                    histogram[r_bin, ori_bin] += magnitude[yi, xi]
    
    # Normalize the histogram
    descriptor = histogram.flatten()
    norm = np.linalg.norm(descriptor)
    if norm > 0:
        descriptor = descriptor / norm
    
    return descriptor, dominant_orientation

def describe_harris_keypoints(image, keypoints):
    """
    Compute rotation-invariant descriptors for all Harris keypoints.
    
    Args:
        image: Grayscale input image
        keypoints: List of keypoint coordinates (y, x)
        
    Returns:
        List of (keypoint, descriptor, orientation) tuples
    """
    results = []
    for kp in keypoints:
        descriptor, orientation = compute_rotation_invariant_descriptor(image, kp)
        results.append((kp, descriptor, orientation))
    
    return results

def match_keypoints(descriptors1, descriptors2, threshold=0.7):
    """
    Match keypoints based on descriptor similarity.
    Uses ratio test for better matching.
    
    Args:
        descriptors1: List of (keypoint, descriptor, orientation) from first image
        descriptors2: List of (keypoint, descriptor, orientation) from second image
        threshold: Ratio threshold for Lowe's ratio test
        
    Returns:
        List of matches (idx1, idx2, distance)
    """
    matches = []
    
    for i, (kp1, desc1, _) in enumerate(descriptors1):
        # Find the two best matches
        distances = []
        for j, (kp2, desc2, _) in enumerate(descriptors2):
            dist = np.linalg.norm(desc1 - desc2)
            distances.append((j, dist))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        # Apply ratio test
        if len(distances) >= 2:
            best_idx, best_dist = distances[0]
            second_best_dist = distances[1][1]
            
            if best_dist < threshold * second_best_dist:
                matches.append((i, best_idx, best_dist))
    
    return matches
'''
# Example usage:
def demo_rotation_invariant_harris():
    # Load an image
    img = cv2.imread('white_rec.jpg', cv2.IMREAD_GRAYSCALE)
    
    # Create a rotated version
    center = (img.shape[1] // 2, img.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
    
    # Detect keypoints in both images
    keypoints_orig = compute_harris_keypoints(img)
    keypoints_rotated = compute_harris_keypoints(rotated_img)
    
    # Compute descriptors
    descriptors_orig = describe_harris_keypoints(img, keypoints_orig)
    descriptors_rotated = describe_harris_keypoints(rotated_img, keypoints_rotated)
    
    # Match keypoints
    matches = match_keypoints(descriptors_orig, descriptors_rotated)
    
    # Sort matches by distance (better matches first)
    matches.sort(key=lambda x: x[2])
    
    # Display results
    print(f"Found {len(matches)} matches between original and rotated image")
    
    # Draw matches (simplified version)
    result_img = np.hstack((img, rotated_img))
    result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
    
    width = img.shape[1]
    for i, j, _ in matches[:20]:  # Show top 20 matches
        pt1 = (int(keypoints_orig[i][1]), int(keypoints_orig[i][0]))
        pt2 = (int(keypoints_rotated[j][1]) + width, int(keypoints_rotated[j][0]))
        cv2.line(result_img, pt1, pt2, (0, 255, 0), 1)
        cv2.circle(result_img, pt1, 3, (0, 0, 255), -1)
        cv2.circle(result_img, pt2, 3, (0, 0, 255), -1)
    
    cv2.imshow("Matches", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()''' 
    
def demo_rotation_invariant_harris(image_path='example.jpg', rotation_angle=45, max_matches=20):
    """
    Demonstrate the rotation-invariant Harris corner detector and descriptor.
    
    Args:
        image_path: Path to the input image
        rotation_angle: Angle to rotate the image (in degrees)
        max_matches: Maximum number of matches to display
    """
    # Load an image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Create a sample image if none is provided
        img = np.zeros((300, 300), dtype=np.uint8)
        # Add some shapes for corners
        cv2.rectangle(img, (50, 50), (100, 100), 255, -1)
        cv2.rectangle(img, (150, 150), (200, 200), 255, -1)
        cv2.circle(img, (200, 75), 25, 255, -1)
    
    # Create a rotated version
    center = (img.shape[1] // 2, img.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
    
    # Detect keypoints in both images
    keypoints_orig = compute_harris_keypoints(img)
    keypoints_rotated = compute_harris_keypoints(rotated_img)
    
    print(f"Found {len(keypoints_orig)} keypoints in original image")
    print(f"Found {len(keypoints_rotated)} keypoints in rotated image")
    
    # Compute descriptors
    descriptors_orig = describe_harris_keypoints(img, keypoints_orig)
    descriptors_rotated = describe_harris_keypoints(rotated_img, keypoints_rotated)
    
    # Match keypoints
    matches = match_keypoints(descriptors_orig, descriptors_rotated)
    
    # Sort matches by distance (better matches first)
    matches.sort(key=lambda x: x[2])
    
    print(f"Found {len(matches)} matches between original and rotated image")
    
    # Visualize original and rotated images with keypoints
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    for y, x in keypoints_orig:
        ax1.plot(x, y, 'ro', markersize=3)
    
    ax2.imshow(rotated_img, cmap='gray')
    ax2.set_title(f'Rotated Image ({rotation_angle}°)')
    ax2.axis('off')
    for y, x in keypoints_rotated:
        ax2.plot(x, y, 'ro', markersize=3)
    
    plt.tight_layout()
    plt.show()
    
    # Visualize matches
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(rotated_img, cmap='gray')
    ax2.set_title(f'Rotated Image ({rotation_angle}°)')
    ax2.axis('off')
    
    # Plot top matches
    for idx, (i, j, dist) in enumerate(matches[:max_matches]):
        y1, x1 = keypoints_orig[i]
        y2, x2 = keypoints_rotated[j]
        
        ax1.plot(x1, y1, 'ro', markersize=5)
        ax2.plot(x2, y2, 'ro', markersize=5)
        
        # Add a line connecting the matches
        con = ConnectionPatch(xyA=(x1, y1), xyB=(x2, y2), 
                             coordsA="data", coordsB="data",
                             axesA=ax1, axesB=ax2, color='green', linewidth=1)
        ax2.add_artist(con)
        
        # Add match number
        ax1.annotate(f"{idx+1}", (x1, y1), xytext=(5, 5), 
                    textcoords='offset points', color='white', fontsize=8)
        ax2.annotate(f"{idx+1}", (x2, y2), xytext=(5, 5), 
                    textcoords='offset points', color='white', fontsize=8)
    
    plt.suptitle(f'Top {min(max_matches, len(matches))} Matches')
    plt.tight_layout()
    plt.show()
    
    # Plot the descriptor for one of the matched keypoints
    if len(matches) > 0:
        best_match_idx = matches[0][0]
        best_kp, best_desc, _ = descriptors_orig[best_match_idx]
        
        # Visualize the descriptor
        descriptor_fig = visualize_descriptor(best_desc)
        plt.suptitle(f'Descriptor for Keypoint {best_match_idx}')
        plt.show()
        
        # Show the keypoint neighborhood
        y, x = best_kp
        max_radius = 12  # Same as used in descriptor
        patch_size = max_radius * 2 + 1
        
        # Extract the patch around the keypoint
        y_min = max(0, y - max_radius)
        y_max = min(img.shape[0], y + max_radius + 1)
        x_min = max(0, x - max_radius)
        x_max = min(img.shape[1], x + max_radius + 1)
        
        patch = np.zeros((patch_size, patch_size), dtype=np.uint8)
        patch[y_min - (y - max_radius):y_max - (y - max_radius), 
              x_min - (x - max_radius):x_max - (x - max_radius)] = img[y_min:y_max, x_min:x_max]
        
        # Show the patch
        plt.figure(figsize=(6, 6))
        plt.imshow(patch, cmap='gray')
        plt.title(f'Neighborhood of Keypoint {best_match_idx}')
        plt.axis('off')
        plt.show()
if __name__== "__main__":
    demo_rotation_invariant_harris()