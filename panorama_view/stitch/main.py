import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

def warpImg(img1_t, img2_t, H_t):
    rows_1, cols_1 = img1_t.shape[:2]
    rows_2, cols_2 = img2_t.shape[:2]

    list_points_1 = np.float32([[0, 0], [0, rows_1], [cols_1, rows_1], [cols_1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0, 0], [0, rows_2], [cols_2, rows_2], [cols_2, 0]]).reshape(-1, 1, 2)

    list_points_2 = cv2.perspectiveTransform(temp_points, H_t)

    list_points = np.concatenate((list_points_1, list_points_2), axis=0)

    [x_min, y_min] = np.int32(list_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2_t, H_translation.dot(H_t), (x_max-x_min, y_max-y_min))
    output_img[translation_dist[1]:rows_1+translation_dist[1], translation_dist[0]:cols_1+translation_dist[0]] = img1_t

    return output_img


def readImgs():
    path = sorted(glob.glob("src/*.jpg"))
    return [cv2.imread(img) for img in path]

def stitching(imgs_t):
    orb = cv2.ORB_create(nfeatures=2000)
    while True:
        print(2)
        img_1 = imgs_t.pop(0)
        img_2 = imgs_t.pop(0)

        kp1, ds1 = orb.detectAndCompute(img_1, None)
        kp2, ds2 = orb.detectAndCompute(img_2, None)

        bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

        matches = bf.knnMatch(ds1, ds2, k=2)

        all_matches = []

        for m, n in matches:
            all_matches.append(m)
        
        good = []

        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good.append(m)
        
        MIN_MATCH_COUNT = 5

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            result = warpImg(img_2, img_1, M)

            imgs_t.insert(0, result)

            if len(imgs_t) == 1:
                break

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result

if __name__ == "__main__":
    imgs = readImgs()
    res = stitching(imgs)
    plt.imshow(res)
    plt.show()

