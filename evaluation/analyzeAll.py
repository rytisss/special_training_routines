import os
import glob
import cv2
import numpy as np
from evaluation.statistics import Statistics
from utilities import get_file_name, gather_image_from_dir

# labels dir (contains images)
label_dir = r'C:\Users\rytis\Desktop\major_review_disertation\test_crop\label/'

# prediction dir (contains folder in which are images)
prediction_dir = r'C:\Users\rytis\Desktop\conventional_clusted/'


def get_prediction_according_to_the_labels(label_path, prediction_dir):
    predicted_images_paths = gather_image_from_dir(prediction_dir)
    label_name = get_file_name(label_path)
    for predicted_image_path in predicted_images_paths:
        prediction_name = get_file_name(predicted_image_path)
        if prediction_name == label_name:
            return predicted_image_path


def analize():
    # get labels
    label_paths = gather_image_from_dir(label_dir)
    label_paths.sort()
    # get all prediction folders
    prediction_folders = glob.glob(prediction_dir + '*/')
    for prediction_folder in prediction_folders:
        prediction_folder_name = os.path.basename(os.path.normpath(prediction_folder))
        # get all images in folder
        prediction_paths = gather_image_from_dir(prediction_folder)
        prediction_paths.sort()

        recallSum = 0.0
        precisionSum = 0.0
        accuracySum = 0.0
        f1Sum = 0.0
        IoUSum = 0.0
        dicsSum = 0.0

        for i, label_path in enumerate(label_paths):
            prediction_path = prediction_paths[i]

            #print(f'{label_path} - {prediction_path}')

            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            _, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
            prediction = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)
            _, prediction = cv2.threshold(prediction, 127, 255, cv2.THRESH_BINARY)

            # do analysis
            tp, fp, tn, fn = Statistics.GetParameters(label, prediction)
            recall = Statistics.GetRecall(tp, fn)
            precision = Statistics.GetPrecision(tp, fp)
            accuracy = Statistics.GetAccuracy(tp, fp, tn, fn)
            f1 = Statistics.GetF1Score(recall, precision)
            IoU = Statistics.GetIoU(label, prediction)
            dice = Statistics.GetDiceCoef(label, prediction)
            # print('Recall: ' + str(recall) + ', Precision: ' + str(precision) + ', accuracy: ' + str(accuracy) + ', f1: ' + str(f1) + ', IoU: ' + str(IoU) + ', Dice: ' + str(dice))

            recallSum = recallSum + recall
            precisionSum = precisionSum + precision
            accuracySum = accuracySum + accuracy
            f1Sum = f1Sum + f1
            IoUSum = IoUSum + IoU
            dicsSum = dicsSum + dice

            # cv2.imshow('Label', label)
            # cv2.imshow('Prediction', prediction)
            # cv2.waitKey(0)

            """
            cv2.imshow('Rendered', renderedImage)
            cv2.imshow('Label', label)
            cv2.imshow('Image', image)
            

            cv2.waitKey(1)
            """

        overallRecall = round(recallSum / float(len(label_paths)), 4)
        overallPrecision = round(precisionSum / float(len(label_paths)), 4)
        overallAccuracy = round(accuracySum / float(len(label_paths)), 4)
        overallF1 = round(f1Sum / float(len(label_paths)), 4)
        overallIoU = round(IoUSum / float(len(label_paths)), 4)
        overallDice = round(dicsSum / float(len(label_paths)), 4)

        print(prediction_folder_name +
              'Recall: ' + str(overallRecall) +
              ', Precision: ' + str(overallPrecision) +
              ', accuracy: ' + str(overallAccuracy) +
              ', f1: ' + str(overallF1) +
              ', IoU: ' + str(overallIoU) +
              ', Dice: ' + str(overallDice))


if __name__ == "__main__":
    analize()
