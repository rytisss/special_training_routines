import os
import cv2
from utilities import gather_image_from_dir, get_file_name
from evaluation.statistics import Statistics

def make_statistics(train_labels_dir, label_prefix, accuracy=1.0):
    """
    Loads all the photos and calculates the TP, FP, TN, FN
    """
    image_paths = gather_image_from_dir(train_labels_dir)
    counter = 0
    total_pixel_count = 0

    positive_pixel_count = 0
    negative_pixel_count = 0

    for image_path in image_paths:
        image_name = get_file_name(image_path)
        label_image = None
        if label_prefix == '':
            # get all images
            label_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            counter = counter + 1
        else:
            if not (label_prefix in image_name):
                continue
            else:
                label_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                counter = counter + 1
        # in case of non 0 or 255 in images
        _, label_image = cv2.threshold(label_image, 127, 255, cv2.THRESH_BINARY)

        # count pixels
        height, width = label_image.shape[:2]
        total_pixel_count = total_pixel_count + height * width

        cv2.namedWindow('label', cv2.WINDOW_NORMAL)
        cv2.imshow('label', label_image)
        cv2.waitKey(1)

        positive_pixels = int(cv2.countNonZero(label_image))
        negative_pixels = height * width - positive_pixels

        positive_pixel_count = positive_pixel_count + positive_pixels
        negative_pixel_count = negative_pixel_count + negative_pixels
    return counter, total_pixel_count, positive_pixel_count, negative_pixel_count


def get_tp_fp_tn_fn(positive_pixels, negative_pixels, recall, precision):
    tp = int(float(positive_pixels) * recall)
    fn = positive_pixels - tp
    fp = int(float(tp) / precision - float(tp))
    tn = negative_pixels - fp
    return tp, fp, tn, fn


if __name__ == "__main__":
    # Crack500
    print('\nCrack500' * 10)
    counter, total_pixel_count, positive_pixel_count, negative_pixel_count = number_of_images = make_statistics(
        r'D:\pavement defect data\crack500\testdata/', '_mask')
    print(f'image counted: {counter}')
    print(f'total: {total_pixel_count}')
    print(f'positive: {positive_pixel_count}')
    print(f'negative: {negative_pixel_count}')
    recalls = {'UNet': 0.7033,
               'ResUNet': 0.7002,
               'ResUNet+ASPP': 0.6944,
               'ResUNet+ASPP+AG': 0.7386,
               'ResUNet+ASPP_WF': 0.7524,
               'ResUNet+ASPP_WF+AG': 0.7829}

    precisions = {'UNet': 0.6996,
                  'ResUNet': 0.7083,
                  'ResUNet+ASPP': 0.7152,
                  'ResUNet+ASPP+AG': 0.6808,
                  'ResUNet+ASPP_WF': 0.6789,
                  'ResUNet+ASPP_WF+AG': 0.6447}

    for i, (configuration, value) in enumerate(recalls.items()):
        tp, fp, tn, fn = get_tp_fp_tn_fn(positive_pixel_count, negative_pixel_count, recalls[configuration], precisions[configuration])
        sum = tp + fp + tn + fn
        print(f'********************** Recalculated stats {configuration} **********************')
        print(f'Accuracy: {Statistics.GetAccuracy(tp, fp, tn, fn)}')
        print(f'Recall: {Statistics.GetRecall(tp, fn)}')
        print(f'Precision: {Statistics.GetPrecision(tp, fp)}')
        recall = Statistics.GetRecall(tp, fn)
        precision = Statistics.GetPrecision(tp, fp)
        print(f'Dice: {Statistics.GetF1Score(recall,precision)}')

    print(50 * '*')
