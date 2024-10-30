import pdb
import glob
import cv2
import os
import numpy as np
import imutils

class PanaromaStitcher:
    def __init__(self):
        self.images = []
        self.homography_matrices = []

    def make_panaroma_for_images_in(self,path):
        self.read_images(path)
        n = len(self.images)
        if n == 2:
            (result) = self.image_stitch([self.images[0], self.images[1]])
        else:
            (result) = self.image_stitch([self.images[n - 2], self.images[n - 1]])
            for i in range(n - 2):
                (result) = self.image_stitch([self.images[n - i - 3], result])

        return result, self.homography_matrices
            

    def read_images(self, path):
        imf = path
        all_images = sorted(glob.glob(imf + os.sep + '*'))
        print('Found {} Images for stitching'.format(len(all_images)))
        images = []
        for img_path in all_images:
            image = cv2.imread(img_path)
            images.append(image)
        n = len(images)
        for i in range(n):
            images[i] = imutils.resize(images[i], width=400)
        for i in range(n):
            images[i] = imutils.resize(images[i], height=400)
        self.images = images
        return

    def image_stitch(self, images, lowe_ratio=0.75, max_Threshold=5, match_status=False):
        imageB, imageA = images
        key_points_A, features_of_A = self.detect_feature_and_keypoints(imageA)
        key_points_B, features_of_B = self.detect_feature_and_keypoints(imageB)


        actual_key_ptsA, actual_key_ptsB, matches = self.match_keypoints(key_points_A, key_points_B, features_of_A, features_of_B, lowe_ratio)

        manual_Homography_matrix = self.homography_ransac(actual_key_ptsA, actual_key_ptsB, max_Threshold)
        self.homography_matrices.append(manual_Homography_matrix)

        result_image = self.warp_perspective(imageA, imageB, manual_Homography_matrix)
        result_image[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        return result_image
    
    def warp_perspective(self, imageA, imageB, Homography):

        combined_width = imageA.shape[1] + imageB.shape[1]
        combined_height = max(imageA.shape[0], imageB.shape[0])
        
        result_image = np.zeros((combined_height, combined_width, imageA.shape[2]), dtype=imageA.dtype)
        
        result_image[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        
        for y in range(combined_height):
            for x in range(combined_width):
                src_pt = np.array([x, y, 1]).reshape(3, 1)
                dst_pt = np.dot(np.linalg.inv(Homography), src_pt)
                dst_pt /= dst_pt[2] 

                src_x, src_y = dst_pt[0, 0], dst_pt[1, 0]

                if 0 <= src_x < imageA.shape[1] and 0 <= src_y < imageA.shape[0]:
                    x0, y0 = int(src_x), int(src_y)
                    dx, dy = src_x - x0, src_y - y0

                    x1 = min(x0 + 1, imageA.shape[1] - 1)
                    y1 = min(y0 + 1, imageA.shape[0] - 1)

                    top_left = imageA[y0, x0]
                    top_right = imageA[y0, x1]
                    bottom_left = imageA[y1, x0]
                    bottom_right = imageA[y1, x1]

                    result_pixel = (
                        (1 - dx) * (1 - dy) * top_left +
                        dx * (1 - dy) * top_right +
                        (1 - dx) * dy * bottom_left +
                        dx * dy * bottom_right
                    )

                    if np.all(result_image[y, x] == 0):
                        result_image[y, x] = result_pixel
                    
        return result_image


    def detect_feature_and_keypoints(self, image):
        descriptors = cv2.SIFT_create()
        (keypoints, features) = descriptors.detectAndCompute(image, None)
        keypoints = np.float32([i.pt for i in keypoints])
        return keypoints, features


    def get_all_possible_matches(self, featuresA, featuresB):
        match_instance = cv2.DescriptorMatcher_create("BruteForce")
        All_Matches = match_instance.knnMatch(featuresA, featuresB, 2)
        return All_Matches


    def get_all_valid_matches(self, AllMatches, lowe_ratio):
        valid_matches = []
        for val in AllMatches:
            if len(val) == 2 and val[0].distance < val[1].distance * lowe_ratio:
                valid_matches.append((val[0].trainIdx, val[0].queryIdx))
        return valid_matches


    def compute_homography(self, pointsA, pointsB, max_Threshold):
        H = cv2.findHomography(pointsA, pointsB, cv2.RANSAC, max_Threshold)
        return H
    
    def normalize_points(self, points):
        mean = np.mean(points, axis=0)
        std = np.std(points)
        T = np.array([[1 / std, 0, -mean[0] / std],
                    [0, 1 / std, -mean[1] / std],
                    [0, 0, 1]])
        normalized_points = np.dot(T, np.vstack((points.T, np.ones((1, points.shape[0])))))
        return normalized_points[:2].T, T
    
    def homography_dlt(self, src_points, dst_points):
        if src_points.shape[0] != dst_points.shape[0] or src_points.shape[0] < 4:
            raise ValueError("There must be at least 4 corresponding points")

        num_points = src_points.shape[0]

        A = []
        for i in range(num_points):
            x, y = src_points[i]
            x_prime, y_prime = dst_points[i]
            
            A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
            A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])

        A = np.array(A)
        U, S, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        H = H / H[-1, -1]
        return H

    def homography_ransac(self, pointsA, pointsB, max_Threshold):

        pointsA = np.asarray(pointsA)
        pointsB = np.asarray(pointsB)

        assert pointsA.shape[0] == pointsB.shape[0], "Number of points in both sets must be the same."
        num_points = pointsA.shape[0]
        N = int(1e3)

        final_H = None
        n_inliers = 0
        inliersA = None
        inliersB = None

        pointsA_homogeneous = np.hstack([pointsA, np.ones((num_points, 1))])
        pointsB_homogeneous = np.hstack([pointsB, np.ones((num_points, 1))])

        for i in range(10):
            random_indices = np.random.choice(num_points, size=4, replace=False)
            random_ptsA = pointsA[random_indices]
            random_ptsB = pointsB[random_indices]

            norm_ptsA, TA = self.normalize_points(random_ptsA)
            norm_ptsB, TB = self.normalize_points(random_ptsB)

            H_norm = self.homography_dlt(norm_ptsA, norm_ptsB)

            H = np.dot(np.linalg.inv(TB), np.dot(H_norm, TA))

            projected_points = np.dot(H, pointsA_homogeneous.T).T

            epsilon = 1e-10
            projected_points[:, 2] = np.where(projected_points[:, 2] == 0, epsilon, projected_points[:, 2])

            projected_points[:, 0] /= projected_points[:, 2]
            projected_points[:, 1] /= projected_points[:, 2]

            errors = np.sqrt((pointsB[:, 0] - projected_points[:, 0]) ** 2 + 
                            (pointsB[:, 1] - projected_points[:, 1]) ** 2)

            inliers = errors < max_Threshold
            inlier_pointsA = pointsA[inliers]
            inlier_pointsB = pointsB[inliers]
            count = np.sum(inliers)

            if count > n_inliers:
                inliersA = inlier_pointsA
                inliersB = inlier_pointsB
                n_inliers = count

        norm_ptsA, TA = self.normalize_points(inliersA)
        norm_ptsB, TB = self.normalize_points(inliersB)  
        H_norm = self.homography_dlt(norm_ptsA, norm_ptsB)
        H = np.dot(np.linalg.inv(TB), np.dot(H_norm, TA))
        return H


    def match_keypoints(self, KeypointsA, KeypointsB, featuresA, featuresB, lowe_ratio):
        all_matches = self.get_all_possible_matches(featuresA, featuresB)
        valid_matches = self.get_all_valid_matches(all_matches, lowe_ratio)

        if len(valid_matches) <= 4:
            return None

        points_A = np.float32([KeypointsA[i] for (_, i) in valid_matches])
        points_B = np.float32([KeypointsB[i] for (i, _) in valid_matches])
        return points_A, points_B, valid_matches
    
