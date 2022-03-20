import cv2
import numpy as np
from utilities import gather_image_from_dir, get_file_name, make_directory
from conventional_methods import kmeans
from EnFCM import EnFCM
from FCM import FCM
from MFCM import MFCM

# Test images directory
test_images = r'D:\straipsniai\straipsnis\dataForTraining_v3_only_epoch\dataForTraining_v3_only_epoch\best_weights\output\interesting parts\images/'
output_dir = r'C:\Users\rytis\Desktop\conventional_to_visualize/'


def predict():
    kmeans_folder = output_dir + 'blurred_kmeans/'
    make_directory(kmeans_folder)

    enhance_fuzzy_cmeans_folder = output_dir + 'blurred_enhanced_fuzzy_cmeans/'
    make_directory(enhance_fuzzy_cmeans_folder)

    fuzzy_cmeans_folder = output_dir + 'blurred_fuzzy_cmeans/'
    make_directory(fuzzy_cmeans_folder)

    modified_fuzzy_cmeans_folder = output_dir + 'blurred_modified_fuzzy_cmeans/'
    make_directory(modified_fuzzy_cmeans_folder)

    otsu_folder = output_dir + 'blurred_otsu/'
    make_directory(otsu_folder)

    #image_folder = output_dir + 'blurred_image/'
    #make_directory(image_folder)

    image_paths = gather_image_from_dir(test_images)
    # Load and predict on all images from directory
    for i, image_path in enumerate(image_paths):
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_name = get_file_name(image_path)

        image = cv2.medianBlur(image, 3)
        #cv2.imwrite(image_folder + image_name + '.png', image)

        # cv2.imshow('image', image)
        # cv2.imshow('blurred', b_image)
        # cv2.waitKey(0)
        # continue

        # kmeans
        kmeans_image = kmeans(image)
        cv2.imwrite(kmeans_folder + image_name + '.png', kmeans_image)
        # cv2.imshow('kmeans', kmeans_image)
        # Enhanced Fuzzy C-Means Algorithm
        cluster = EnFCM(image, image_bit=8, n_clusters=2, m=2,
                        neighbour_effect=0.2, epsilon=0.1, max_iter=100,
                        kernel_size=3)
        cluster.form_clusters()
        result = np.uint8(cluster.result)
        result[result == 1] = 255
        cv2.imwrite(enhance_fuzzy_cmeans_folder + image_name + '.png', result)

        # cv2.imshow('Enhanced Fuzzy C-Means Algorithm', result)
        # Standard Fuzzy C-Means Algorithm
        cluster = FCM(image, image_bit=8, n_clusters=2, m=2, epsilon=0.02,
                      max_iter=100)
        cluster.form_clusters()
        result = np.uint8(cluster.result)
        result[result == 1] = 255
        cv2.imwrite(fuzzy_cmeans_folder + image_name + '.png', result)

        # cv2.imshow('Fuzzy C-Means Algorithm', result)
        # Modified Fuzzy C-Means Algorithm
        # cluster = MFCM(image, image_bit=8, n_clusters=2, m=2,
        #                neighbour_effect=0.2, epsilon=0.1, max_iter=100,
        #                kernel_size=3)
        # cluster.form_clusters()
        # result = np.uint8(cluster.result)
        # result[result == 1] = 255
        # cv2.imwrite(modified_fuzzy_cmeans_folder + image_name + '*.png', result)
        #
        # cv2.imshow('Modified Fuzzy C-Means Algorithm', result)

        # OTSU
        _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
        cv2.imwrite(otsu_folder + image_name + '.png', otsu)

        # cv2.imshow('OTSU', otsu)
        # adaptive_th_meanc = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
        #                                           cv2.THRESH_BINARY, 15, 2)
        # cv2.imshow('adaptive_th_meanc', adaptive_th_meanc)
        # adaptive_th_gaussianc = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
        #                                               cv2.THRESH_BINARY, 15, 2)
        # cv2.imshow('adaptive_th_gaussianc', adaptive_th_gaussianc)

        # cv2.imshow("image", image)
        # cv2.waitKey(0)


if __name__ == '__main__':
    predict()
