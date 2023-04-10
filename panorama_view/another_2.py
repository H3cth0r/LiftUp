import cv2
import numpy as np

# Load images from multiple cameras
img1 = cv2.imread('stitch/src_three/img(2).jpg')
img2 = cv2.imread('stitch/src_three/img(1).jpg')
img3 = cv2.imread('stitch/src_eight/img(1).png')

# Create Stitcher object
stitcher = cv2.Stitcher_create()


# Stitch images
status, result = stitcher.stitch([img1, img2])

# Display the result
cv2.imshow('Panorama', result)
cv2.waitKey()