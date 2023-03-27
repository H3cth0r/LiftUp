import cv2
import numpy as np
import glob

# Load images from multiple cameras
img1 = cv2.imread('stitch/src_eight/img(3).png')
img2 = cv2.imread('stitch/src_eight/img(2).png')
img3 = cv2.imread('stitch/src_eight/img(3).png')

# Resize images to reduce computational cost
height, width = img1.shape[:2]
max_dim = max(height, width)
scale = 1024 / max_dim
img1 = cv2.resize(img1, (int(width*scale), int(height*scale)))
img2 = cv2.resize(img2, (int(width*scale), int(height*scale)))

# Extract keypoints and descriptors using ORB
orb = cv2.ORB_create(nfeatures=10000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

des1 = des1.astype(np.float32)
des2 = des2.astype(np.float32)

# Match keypoints using FLANN
flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})
matches = flann.knnMatch(des1, des2, k=2)

# Filter good matches using Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good_matches.append(m)

# Estimate homography using RANSAC
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp image 1 using the homography
img1_warped = cv2.warpPerspective(img1, H, (img1.shape[1] + img2.shape[1], img2.shape[0]))

# Combine the two images
result = np.zeros_like(img1_warped)
result[:img2.shape[0], :img2.shape[1]] = img2
result = cv2.addWeighted(img1_warped, 0.5, result, 0.5, 0)

# Perspective correction
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
max_cnt = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(max_cnt)
result = result[y:y+h, x:x+w]

# Display the result
cv2.imshow('Panorama', result)
cv2.waitKey()