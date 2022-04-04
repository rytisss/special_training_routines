import glob
import os

import cv2
from utilities import gather_image_from_dir
from evaluation.statistics import Statistics

data_dir = r'D:\DAGM/'

results_dir = r'D:\dagm_results/'

# train
train_images_dir = 'data/image/'
train_labels_dir = 'data/label/'
# test
test_images_dir = 'data/image/'
test_labels_dir = 'data/label/'



def eval():
    kernels = [8, 16, 32]
    layers = [2, 3, 4, 5]
    # gather all the classes
    classes_folders = glob.glob(data_dir + '*/')
    for classes_folder in classes_folders:
        for kernel in kernels:
            for layer in layers:
                basefolder = os.path.basename(os.path.normpath(classes_folder))
                print(50 * '*')
                print(f'Class {basefolder} - k{kernel}l{layer}')
                print(50 * '*')

                results_folder = f'{results_dir}/{basefolder}/k{kernel}l{layer}/'
                results_folder = glob.glob(results_folder + '*/')[0]
                predictions = gather_image_from_dir(results_folder)
                labels_dir = f'{data_dir}/{basefolder}/Test/label/'
                labels = gather_image_from_dir(labels_dir)

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

                for i, label_path in enumerate(predictions):
                    prediction = cv2.imread(predictions[i], cv2.IMREAD_GRAYSCALE)
                    _, prediction = cv2.threshold(prediction, 127, 255, cv2.THRESH_BINARY)
                    label = cv2.imread(labels[i], cv2.IMREAD_GRAYSCALE)
                    _, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)

                    # if (cv2.countNonZero(label) == 0):
                    #      continue

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
                g = 4




if __name__ == "__main__":
    eval()
