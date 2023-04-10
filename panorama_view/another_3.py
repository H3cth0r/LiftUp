import cv2
import numpy as np
import glob

# Load images from multiple cameras
img1 = cv2.imread('stitch/src_eight/img(3).png')
img2 = cv2.imread('stitch/src_eight/img(2).png')
#img1 = cv2.imread('stitch/src_three/img(2).jpg')
#img2 = cv2.imread('stitch/src_three/img(1).jpg')
img3 = cv2.imread('stitch/src_eight/img(1).png')

# Resize images to reduce computational cost
height, width = img1.shape[:2]
max_dim = max(height, width)
scale = 1024 / max_dim
img1 = cv2.resize(img1, (int(width*scale), int(height*scale)))
img2 = cv2.resize(img2, (int(width*scale), int(height*scale)))
img3 = cv2.resize(img3, (int(width*scale), int(height*scale)))

# Extract keypoints and descriptors using ORB
orb = cv2.ORB_create(nfeatures=10000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
kp3, des3 = orb.detectAndCompute(img3, None)

# Convert descriptors to np.float32 data type
des1 = des1.astype(np.float32)
des2 = des2.astype(np.float32)
des3 = des3.astype(np.float32)

# Match keypoints using FLANN
flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})
matches1_2 = flann.knnMatch(des1, des2, k=2)
matches2_3 = flann.knnMatch(des2, des3, k=2)

# Filter good matches using Lowe's ratio test
good_matches1_2 = []
for m, n in matches1_2:
    if m.distance < 0.7*n.distance:
        good_matches1_2.append(m)
        
good_matches2_3 = []
for m, n in matches2_3:
    if m.distance < 0.7*n.distance:
        good_matches2_3.append(m)

# Estimate homographies using RANSAC
src_pts1_2 = np.float32([kp1[m.queryIdx].pt for m in good_matches1_2]).reshape(-1, 1, 2)
dst_pts1_2 = np.float32([kp2[m.trainIdx].pt for m in good_matches1_2]).reshape(-1, 1, 2)
H1_2, _ = cv2.findHomography(src_pts1_2, dst_pts1_2, cv2.RANSAC, 5.0)

src_pts2_3 = np.float32([kp2[m.queryIdx].pt for m in good_matches2_3]).reshape(-1, 1, 2)
dst_pts2_3 = np.float32([kp3[m.trainIdx].pt for m in good_matches2_3]).reshape(-1, 1, 2)
H2_3, _ = cv2.findHomography(src_pts2_3, dst_pts2_3, cv2.RANSAC, 5.0)

# Warp images using the homographies
img1_warped = cv2.warpPerspective(img1, H1_2, (img1.shape[1] + img2.shape[1], img2.shape[0]))
img3_warped = cv2.warpPerspective(img3, H2_3, (img3.shape[1] + img2.shape[1], img2.shape[0]))

# Combine the three images
result = np.zeros_like(img1_warped)
result[:img2.shape[0], :img2.shape[1]] = img2
result = cv2.addWeighted(img1_warped, 0.5, result, 0.5, 0)
result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

# Create a canvas to hold the stitched images
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
h3, w3 = img3.shape[:2]
max_h = max(h1, h2, h3)
total_w = w1 + w2 + w3  # modified line
result_final = np.zeros((max_h, total_w, 3), dtype=np.uint8)

# Place the first image in the canvas
result_final[:h1, :w1, :] = img1

# Place the second image in the canvas
result_final[:h2, w1:w1 + w2, :] = img2

# Place the third image in the canvas
result_final[:h3, w1 + w2:, :] = img3

# Blend images together using multi-band blending
blend_width = 20
multi_band_blender = cv2.detail_MultiBandBlender()
multi_band_blender.setNumBands(5)
#multi_band_blender.setWeightSharpness(0.1)
#multi_band_blender.setSharpness(0.1)
multi_band_blender.feed(result_final.astype(np.float32), np.zeros_like(result_final), np.ones_like(result_final[:, :, 0]).astype(np.uint8))
multi_band_blender.feed(result.astype(np.float32), (np.abs(result - result_final) > blend_width).astype(np.uint8), np.zeros_like(result[:, :, 0]).astype(np.uint8))
result_blend = multi_band_blender.blend(np.zeros((result_final.shape[0], result_final.shape[1], 1), dtype=np.float32))
result_blend = np.clip(result_blend, 0, 255).astype(np.uint8)

# Show final result
cv2.imshow('Panorama', result_blend)
cv2.waitKey()
cv2.destroyAllWindows()