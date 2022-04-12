import os
import glob
import cv2
import numpy as np
from statistics import Statistics
from scipy.stats import sem


def CheckIfPixelInSpecificPlaceWithinTolerance(x, y, label, prediction, tolerance):
    height, width = label.shape
    # basically take square with offset 'tolerance'. Also check if we are not out of image range

    # value of a particular pixel in label
    label_val = label[y, x]
    # store all value in range
    pixels_values = []
    left = x - tolerance
    right = x + tolerance + 1  # +1, cause last index in range is not evaluated
    top = y - tolerance
    bottom = y + tolerance + 1  # +1, cause last index in range is not evaluated

    visualize = False
    visualize_matrix = np.zeros((width, height), np.uint8)

    # if visualize:
    # print('Current pixel: ' + str(x) + ', ' + str(y))

    for x_ in range(left, right):
        for y_ in range(top, bottom):
            # check if pixel is within image
            if x_ >= 0 and x_ < width and y_ >= 0 and y_ < height:
                prediction_val = prediction[y_, x_]
                if prediction_val != 0 and prediction_val != 255:
                    print('Something is wrong!')
                pixels_values.append(prediction_val)
                if visualize:
                    # print(str(x_) + ', ' + str(y_))
                    visualize_matrix[x_, y_] = 255
                    cv2.imshow('visual', visualize_matrix)
                    cv2.waitKey(1)
                if prediction_val == label_val:
                    return True  # found
    # cv2.waitKey(1000)
    return False  # nothing is found


def AnalyzeSample(label, prediction):
    height, width = label.shape
    # tolerance in pixel how far another 'possible' pixel can be
    tolerance = 2
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for y in range(0, height):
        for x in range(0, width):
            res = CheckIfPixelInSpecificPlaceWithinTolerance(x, y, label, prediction, tolerance)
            # check the value to know what what we are looking for: positives or negatives
            if label[y, x] == 0:
                # negative
                if res:
                    tn += 1
                else:
                    fn += 1
            else:
                if res:
                    tp += 1
                else:
                    fp += 1
    # print('True positive: ' + str(tp))
    # print('False positive: ' + str(fp))
    # print('True negative: ' + str(tn))
    # print('False negative: ' + str(fn))

    return tp, fp, tn, fn


def AnalyzePredictions():
    imagePaths = glob.glob(r'D:\pavement\GAPs384\Test\Images/' + '*.jpg')
    labelPaths = glob.glob(r'D:\pavement\GAPs384\Test\Labels/' + '*.jpg')
    predictionImagePaths = glob.glob(r'C:\Users\rytis\Desktop\major_review_disertation\pavement_all\pretrained_UNet4_res_aspp_AG\gaps384/' + '*.jpg')
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
    counter = 0
    for i in range(0, len(predictionImagePaths)):
        image = cv2.imread(imagePaths[i], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(labelPaths[i], cv2.IMREAD_GRAYSCALE)
        _, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
        prediction = cv2.imread(predictionImagePaths[i], cv2.IMREAD_GRAYSCALE)
        _, prediction = cv2.threshold(prediction, 127, 255, cv2.THRESH_BINARY)
        cv2.imshow('image', image)
        cv2.imshow('label', label)
        cv2.imshow('prediction', prediction)
        cv2.waitKey(1)

        tp, fp, tn, fn = AnalyzeSample(label, prediction)

        mcc = Statistics.GetMCC(tp, fp, tn, fn)
        recall = Statistics.GetRecall(tp, fn)
        precision = Statistics.GetPrecision(tp, fp)
        dice = Statistics.GetF1Score(recall, precision)
        accuracy = Statistics.GetAccuracy(tp, fp, tn, fn)

        accuracy_array.append(accuracy)
        precision_array.append(precision)
        recall_array.append(recall)
        mcc_array.append(mcc)
        dice_array.append(dice)
        tp_array.append(tp)
        fp_array.append(fp)
        tn_array.append(tn)
        fn_array.append(fn)

    sum_recall = float(sum(recall_array))
    sum_precision = float(sum(precision_array))
    sum_accuracy = float(sum(accuracy_array))
    sum_f1 = float(sum(f1_array))
    sum_IoU = float(sum(IoU_array))
    sum_dice = float(sum(dice_array))
    sum_mcc = float(sum(mcc_array))

    pic_count = float(len(predictionImagePaths))  # float(len(label_paths))
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

    print(f'acc: {avg_accuracy}, acc_mean: {sem(accuracy_array)}')
    print(f'recall: {avg_recall}, recall_mean: {sem(recall_array)}')
    print(f'precision: {avg_precision}, precision_mean: {sem(precision_array)}')
    # print(f'f1: {avg_f1}')
    # print(f'IoU: {avg_IoU}, IoU_mean: {sem(IoU_array)}')
    print(f'dice: {avg_dice}, Dice_mean: {sem(dice_array)}')
    print(f'mcc: {avg_mcc}, MCC_mean: {sem(mcc_array)}')
    print(f'tp: {tp_sum}')
    print(f'fp: {fp_sum}')
    print(f'tn: {tn_sum}')
    print(f'fn: {fn_sum}')


def main():
    AnalyzePredictions()


if __name__ == "__main__":
    main()
