
import cv2
import numpy as np
import math

# Load images from multiple cameras
img = cv2.imread('stitch/src_eight/img(2).png')

# Define image dimensions
img_height, img_width, channels = img.shape

# Define camera FOV in degrees
fov_degrees = 60

# Convert FOV to radians
fov_radians = math.radians(fov_degrees)

# Estimate focal length
focal_length = img_width / (2 * math.tan(fov_radians / 2))

# Define camera matrix and distortion coefficients
K = np.array([[focal_length, 0, img.shape[1]/2],
              [0, focal_length, img.shape[0]/2],
              [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

# Fix barrel distortion
img_undistorted = cv2.undistort(img, K, dist_coeffs)

# Display the results
cv2.imshow('Original Image', img)
cv2.imshow('Undistorted Image', img_undistorted)
cv2.waitKey(0)