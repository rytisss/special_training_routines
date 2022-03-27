import os
import cv2
import glob
import numpy as np
from utilities import gather_image_from_dir, get_file_name
from evaluation.statistics import Statistics
from scipy.stats import sem
from statsmodels.stats.weightstats import ztest as ztest

if __name__ == "__main__":
    label_dir = r'C:\Users\rytis\Desktop\major_review_disertation\test_crop_corners\label/'
    prediction_dir = r'C:\Users\rytis\Desktop\major_review_disertation\test_crop_corners\cnn_predictions/'

    label_paths = gather_image_from_dir(label_dir)
    label_paths.sort()

    prediction_cnn_dirs = glob.glob(prediction_dir + '*/')
    architecture_count = len(prediction_cnn_dirs)

    each_architecture_results = {}

    architecture_dice_values = {}

    for architecture_index, architecture_prediction_folder in enumerate(prediction_cnn_dirs):
        print(50 * '*')
        tp_array = []
        fp_array = []
        tn_array = []
        fn_array = []

        recall_array = []
        precision_array = []
        accuracy_array = []
        f1_array = []
        IoU_array = []
        dice_array = []
        mcc_array = []

        predicted_images = gather_image_from_dir(architecture_prediction_folder)
        predicted_images.sort()

        counter = 0
        architecture_name = os.path.basename(os.path.normpath(architecture_prediction_folder))

        architecture_images = {}
        if not (architecture_name == 'UNet4_res_aspp_SE'):
            continue

        print(architecture_name)
        for i, label_path in enumerate(label_paths):
            prediction = cv2.imread(predicted_images[i], cv2.IMREAD_GRAYSCALE)
            _, prediction = cv2.threshold(prediction, 127, 255, cv2.THRESH_BINARY)
            label = cv2.imread(label_paths[i], cv2.IMREAD_GRAYSCALE)
            _, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)

            if (cv2.countNonZero(label) > 0):
                prediction_original = prediction.copy()
                kernel = np.ones((3, 3), np.uint8)
                erode_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
                prediction = cv2.dilate(prediction, erode_kernel, iterations=1)
                # erode_kernel = np.array([[0, 1, 1], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
                # prediction = cv2.erode(prediction, erode_kernel, iterations=1)
                erode_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=np.uint8)
                prediction = cv2.erode(prediction, erode_kernel, iterations=2)
                # erode_kernel = np.array([[0, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=np.uint8)
                # prediction = cv2.dilate(prediction, erode_kernel, iterations=1)
                # prediction = cv2.dilate(prediction, erode_kernel, iterations=1)
                # prediction = cv2.dilate(prediction, erode_kernel, iterations=1)
                # prediction = cv2.erode(prediction, erode_kernel, iterations=1)
                # cv2.imshow('original', prediction_original)
                # cv2.imshow('dilation', prediction)
                # cv2.waitKey(1)

            # if (cv2.countNonZero(label) == 0):
            #    continue
            file_name = get_file_name(predicted_images[i])
            architecture_images.update({file_name: prediction})

            tp, fp, tn, fn = Statistics.GetParameters(label, prediction)
            recall = Statistics.GetRecall(tp, fn)
            precision = Statistics.GetPrecision(tp, fp)
            accuracy = Statistics.GetAccuracy(tp, fp, tn, fn)
            f1 = Statistics.GetF1Score(recall, precision)
            IoU = Statistics.GetIoU(label, prediction)
            dice = Statistics.GetDiceCoef(label, prediction)
            mcc = Statistics.GetMCC(tp, fp, tn, fn)

            tp_array.append(tp)
            fp_array.append(fp)
            tn_array.append(tn)
            fn_array.append(fn)

            recall_array.append(recall)
            precision_array.append(precision)
            accuracy_array.append(accuracy)
            f1_array.append(f1)
            IoU_array.append(IoU)
            dice_array.append(dice)
            mcc_array.append(mcc)

            counter = counter + 1

        sum_recall = float(sum(recall_array))
        sum_precision = float(sum(precision_array))
        sum_accuracy = float(sum(accuracy_array))
        sum_f1 = float(sum(f1_array))
        sum_IoU = float(sum(IoU_array))
        sum_dice = float(sum(dice_array))
        sum_mcc = float(sum(mcc_array))

        pic_count = float(counter)  # float(len(label_paths))
        avg_recall = sum_recall / pic_count
        avg_precision = sum_precision / pic_count
        avg_accuracy = sum_accuracy / pic_count
        avg_f1 = sum_f1 / pic_count
        avg_IoU = sum_IoU / pic_count
        avg_dice = sum_dice / pic_count
        avg_mcc = sum_mcc / pic_count

        tp_sum = sum(tp_array)
        fp_sum = sum(fp_array)
        tn_sum = sum(tn_array)
        fn_sum = sum(fn_array)

        print(f'acc: {avg_accuracy}')
        print(f'recall: {avg_recall}')
        print(f'precision: {avg_precision}')
        print(f'f1: {avg_f1}')
        print(f'IoU: {avg_IoU}')
        print(f'dice: {avg_dice}')
        print(f'mcc: {avg_mcc}')
        print(f'tp: {tp_sum}')
        print(f'fp: {fp_sum}')
        print(f'tn: {tn_sum}')
        print(f'fn: {fn_sum}')
        print('---------------')
        print(f'acc_mean: {sem(accuracy_array)}')
        print(f'recall_mean: {sem(recall_array)}')
        print(f'precision_mean: {sem(precision_array)}')
        print(f'f1_mean: {sem(f1_array)}')
        print(f'IoU_mean: {sem(IoU_array)}')
        print(f'Dice_mean: {sem(dice_array)}')
        print(f'MCC_mean: {sem(mcc_array)}')

        each_architecture_results.update({architecture_name: architecture_images})

        architecture_dice_values.update({architecture_name: dice_array})

        # cv2.imshow('label', label)
        # cv2.imshow('prediction', prediction)
        # cv2.waitKey(1)

    # print(ztest(significance_dice_array[0], significance_dice_array[1], value=0))
    for key, value in each_architecture_results.items():
        if os.path.exists(key + '/'):
            print(f'Path {key} exist, skip')
        else:
            os.makedirs(key)
            for photo_name, photo_array in value.items():
                cv2.imwrite(f'{key}/{photo_name}.png', photo_array)
    g = 1
