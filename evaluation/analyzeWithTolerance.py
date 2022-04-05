import os
import glob
import cv2
import numpy as np
from statistics import Statistics


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
    tolerance = 5
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
    print('True positive: ' + str(tp))
    print('False positive: ' + str(fp))
    print('True negative: ' + str(tn))
    print('False negative: ' + str(fn))
    mcc = Statistics.GetMCC(tp, fp, tn, fn)
    recall = Statistics.GetRecall(tp, fn)
    precision = Statistics.GetPrecision(tp, fp)
    dice = Statistics.GetF1Score(recall, precision)
    return mcc, dice


def AnalyzePredictions():
    imagePaths = glob.glob(r'D:\pavement\crack500\Test\Images/' + '*.jpg')
    labelPaths = glob.glob(r'D:\pavement\crack500\Test\Labels/' + '*.jpg')
    predictionImagePaths = glob.glob(r'C:\Users\rytis\Desktop\major_review_disertation\pavement_all\pretrained_Unet4_res_asppWF\crack500/' + '*.jpg')
    mccs = []
    dices = []
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
        mcc, dice = AnalyzeSample(label, prediction)
        mccs.append(mcc)
        dices.append(dice)
    mcc_average = float(sum(mccs)) / float(len(mccs))
    dice_average = float(sum(dices)) / float(len(dices))
    print(mccs)
    print(f'Average MCC: {mcc_average}')
    print(dices)
    print(f'Average Dice: {dice_average}')


def main():
    AnalyzePredictions()


if __name__ == "__main__":
    main()
