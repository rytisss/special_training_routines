import cv2
import numpy as np
import math


class Augmentation:
    @staticmethod
    def RotateImage(image, angle):
        if angle == 0.0:
            rotated_mat = image.copy()
            return rotated_mat
        elif angle == 90.0:
            rotated_mat = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return rotated_mat
        elif angle == 180.0:
            rotated_mat = cv2.rotate(image, cv2.ROTATE_180)
            return rotated_mat
        elif angle == 270.0:
            rotated_mat = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            return rotated_mat
        else:
            height, width = image.shape[:2]
            image_center = (width / 2, height / 2)

            rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

            abs_cos = abs(rotation_mat[0, 0])
            abs_sin = abs(rotation_mat[0, 1])

            bound_w = int(height * abs_sin + width * abs_cos)
            bound_h = int(height * abs_cos + width * abs_sin)

            rotation_mat[0, 2] += bound_w / 2 - image_center[0]
            rotation_mat[1, 2] += bound_h / 2 - image_center[1]

            rotated_mat = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))
            return rotated_mat

    @staticmethod
    def FlipImageHorizontally(image):
        flipped_image = cv2.flip(image, 0)
        return flipped_image

    @staticmethod
    def BrighnessCorrection(image, beta):
        adjusted_image = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
        return adjusted_image

    @staticmethod
    def Adjust_gamma(image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    @staticmethod
    def AddNoise(image):
        gaussian_noise = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.randn(gaussian_noise, 20, 15)
        gaussian_noise = (gaussian_noise * 0.5).astype(np.uint8)
        noisy_image = cv2.add(image, gaussian_noise)
        return noisy_image

    @staticmethod
    def RotatePoint(pt, rotation_pt, angle):
        # normlaize point
        norm_x = pt[0] - rotation_pt[0]
        norm_y = pt[1] - rotation_pt[1]

        sin_a = math.sin(angle * math.pi / 180.0)
        cos_a = math.cos(angle * math.pi / 180.0)

        pt_rot_x = norm_x * cos_a - norm_y * sin_a + rotation_pt[0]
        pt_rot_y = norm_x * sin_a + norm_y * cos_a + rotation_pt[1]

        return (pt_rot_x, pt_rot_y)
