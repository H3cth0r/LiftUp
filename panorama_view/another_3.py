import cv2
import numpy as np

# Load images
img1 = cv2.imread('stitch/src_eight/img(3).png')
img2 = cv2.imread('stitch/src_eight/img(2).png')

# Load homography from file
H = np.loadtxt('homography.txt')

# Warp second image using homography
result = cv2.warpPerspective(img2, H, (img1.shape[1]+img2.shape[1], img1.shape[0]))

# Blend images together
result[:img1.shape[0], :img1.shape[1]] = img1
result[:img2.shape[0], img1.shape[1]:] = img2

# Display the result
cv2.imshow('Panorama', result)
cv2.waitKey()