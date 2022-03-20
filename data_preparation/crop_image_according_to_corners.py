import os
import glob
import cv2
import numpy as np
from evaluation.statistics import Statistics
from utilities import get_file_name, gather_image_from_dir, get_file_name_with_ext, make_directory
from processing import split_image_to_tiles, crop_image_from_region

# image dir (contains images)
image_dir = r'D:\straipsniai\straipsnis\test\Data_with_gamma_correction\Image/'

# label dir (contains images)
label_dir = r'D:\straipsniai\straipsnis\test\Data_with_gamma_correction\Label/'

# output
output_dir = r'C:\Users\rytis\Desktop\major_review_disertation\test_crop_corners/'


def make_test():
    full_images = gather_image_from_dir(image_dir)
    full_labels = gather_image_from_dir(label_dir)

    crop_image_dir = output_dir + 'image/'
    crop_label_dir = output_dir + 'label/'
    make_directory(crop_image_dir)
    make_directory(crop_label_dir)

    # corners of object in photos
    corners_list = {
        '2[800]': ((82, 326), (1915, 319), (1910, 2450), (78, 2457)),
        '3[600]': ((101, 240), (1926, 233), (1921, 3286), (93, 3294)),
        '5[800]': ((113, 593), (1937, 616), (1898, 2744), (73, 2723)),
        '7[800]': ((111, 149), (1943, 141), (1939, 5052), (107, 5064)),
        '8[600]': ((115, 86), (1941, 71), (1949, 4980), (123, 4996)),
        '12[800]': ((128, 148), (1962, 148), (1926, 5048), (93, 5048)),
        'Image__2018-07-12__16-51-15': ((1935, 631), (3969, 554), (4076, 3875), (2042, 3953)),
        'Image__2018-07-12__16-51-29': ((2034, 559), (4064, 658), (3889, 3979), (1859, 3886)),
        'Image__2018-07-12__16-57-53': ((1949, 128), (3981, 33), (4115, 3356), (2089, 3449)),
        'Image__2018-07-12__17-17-13': ((899, 454), (2937, 381), (3098, 5676), (1064, 5749)),
        'Image__2018-07-12__17-17-50': ((962, 644), (3001, 637), (2984, 5955), (947, 5962)),
        'Image__2018-07-12__17-18-43': ((960, 744), (2995, 739), (2977, 6043), (939, 6051)),
        'Image__2018-07-12__17-26-40': ((1410, 1023), (2309, 886), (2560, 2627), (1656, 2757)),
        'Image__2018-08-02__11-07-50': ((243, 801), (3034, 815), (3024, 4198), (248, 4178)),
        'IMG_2020-09-24_0816396709': ((346, 218), (1624, 212), (1620, 1647), (345, 1653)),
        'IMG_2020-09-24_1039480289': ((349, 209), (1071, 207), (1065, 1641), (345, 1644)),
        'IMG_2020-10-13_1525072797': ((347, 227), (1561, 222), (1555, 1262), (345, 1262)),
        'IMG_2020-10-14_1705340582': ((347, 221), (1403, 218), (1397, 1453), (344, 1455)),
        'IMG_0953521980_gamma12': ((455, 544), (1892, 534), (1902, 2448), (464, 2457)),
        'IMG_1054528895_gamma12': ((221, 293), (1267, 283), (1270, 900), (221, 909)),
        'IMG_1137073268_gamma12': ((223, 294), (1268, 285), (1270, 903), (221, 910)),
        'IMG_1202544992': ((1022, 923), (2451, 930), (2431, 2362), (1019, 2344)),
        'IMG_1458458410': ((348, 192), (2426, 196), (2419, 1809), (344, 1806)),
        'IMG_1619139734_gamma12': ((125, 71), (799, 71), (800, 801), (125, 795))
    }

    for data_sample in zip(full_images, full_labels):
        image_path = data_sample[0]
        label_path = data_sample[1]
        # print('Picture number: ' + str(counter))
        # print('Image: ' + image_path)
        # print('Label: ' + label_path)

        # get corners
        image_name = get_file_name(image_path)
        corners = corners_list.get(image_name)

        # open images
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # get bounding box
        bbox = cv2.boundingRect(np.asarray(corners))

        # remove artifacts, make label 0 or 255 only
        _, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)

        image = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        label = label[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        image_height, image_width = image.shape[:2]

        rois = split_image_to_tiles(image_height, image_width, 320, 320, 160, 160)

        cropped_image_output_dir = output_dir + 'image/'
        cropped_label_output_dir = output_dir + 'label/'

        os.makedirs(cropped_image_output_dir, exist_ok=True)
        os.makedirs(cropped_label_output_dir, exist_ok=True)

        for roi in rois:
            x = roi[0]
            y = roi[1]
            w = roi[2]
            h = roi[3]

            roi_ = [roi[0], roi[1], roi[0] + 320, roi[1] + 320]
            cropped_image = crop_image_from_region(image, roi_)
            cropped_label = crop_image_from_region(label, roi_)
            # cv2.imshow('image', cropped_image)
            # cv2.imshow('label', cropped_label)

            name = image_name + '_' + str(x) + '_' + str(y)
            cv2.imwrite(cropped_image_output_dir + name + '.png', cropped_image)
            cv2.imwrite(cropped_label_output_dir + name + '.png', cropped_label)

            # cv2.waitKey(1)



if __name__ == "__main__":
    make_test()
