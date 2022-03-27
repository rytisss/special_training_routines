import glob
import os
import cv2
from utilities import gather_image_from_dir, get_file_name
from evaluation.statistics import Statistics
from scipy.stats import sem
import Orange
import matplotlib.pyplot as plt

from statsmodels.stats.weightstats import ztest as ztest

if __name__ == "__main__":
    dataset_dir = r'C:\Users\rytis\Desktop\major_review_disertation\pavement_datasets/'
    prediction_dir = r'C:\Users\rytis\Desktop\major_review_disertation\pavement_all/'
    datasets = ['crack500', 'CrackForest', 'GAPs384']
    architectures = glob.glob(prediction_dir + '*/')

    architecture_dice_values = {}

    for dataset in datasets:
        dataset_dict = {}
        for architecture in architectures:
            print(50 * '*')
            prediction_folder = architecture + dataset + '/'
            print(prediction_folder)
            predicted_images = gather_image_from_dir(prediction_folder)

            architecture_name = os.path.basename(os.path.dirname(architecture))

            predicted_images.sort()
            labels = gather_image_from_dir(dataset_dir + dataset + '/Test/Labels/')
            labels.sort()
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

            for i, label_path in enumerate(labels):
                prediction = cv2.imread(predicted_images[i], cv2.IMREAD_GRAYSCALE)
                _, prediction = cv2.threshold(prediction, 127, 255, cv2.THRESH_BINARY)
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
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
            print(f'acc_mean: {sem(accuracy_array)}')
            print(f'recall_mean: {sem(recall_array)}')
            print(f'precision_mean: {sem(precision_array)}')
            print(f'f1_mean: {sem(f1_array)}')
            print(f'IoU_mean: {sem(IoU_array)}')
            print(f'Dice_mean: {sem(dice_array)}')
            print(f'MCC_mean: {sem(mcc_array)}')

            dataset_dict.update({architecture_name: dice_array})

            # cv2.imshow('label', label)
            # cv2.imshow('prediction', prediction)
            # cv2.waitKey(1)
        architecture_dice_values.update({dataset: dataset_dict})

    for architecture_name, architecture_results in architecture_dice_values.items():
        best_architecture = ''
        if 'GAPs384' == architecture_name:
            best_architecture = 'pretrained_UNet4_res_aspp_AG'
        elif 'CrackForest' == architecture_name:
            best_architecture = 'pretrained_Unet4_res_asppWF'
        elif 'crack500' == architecture_name:
            best_architecture = 'pretrained_Unet4_res_asppWF_AG'
        else:
            print('Error!')
        print(50 * '*')
        print(architecture_name)
        print(50 * '*')
        for name, values in architecture_results.items():
            if name == best_architecture:
                continue
            z_test_significance = ztest(architecture_results[best_architecture], architecture_results[name],
                                        value=0)
            print(f'Ztest {best_architecture} to {name} {z_test_significance}')

    architectures = ['UNet', 'ResUNet', 'ResUNet+ASPP', 'ResUNet+ASPP+AG', 'ResUNet+ASPP_WF', 'ResUNet+ASPP_WF+AG']
    avg_rank = [5.6666, 4.6666, 2.3333, 3.0, 2.0, 3.3333]

    # bonferroni-dunn
    cd = Orange.evaluation.compute_CD(avg_rank, 3, alpha='0.05', test='bonferroni-dunn') #tested on 3 datasets
    print('cd', cd)

    Orange.evaluation.graph_ranks(avg_rank, architectures, cd=cd, width=5, textspace=1.5, cdmethod=0)
    plt.show()

    # Nemenyi
    cd = Orange.evaluation.compute_CD(avg_rank, 3, alpha='0.05') #tested on 3 datasets
    print('cd', cd)

    Orange.evaluation.graph_ranks(avg_rank, architectures, cd=cd, width=5, textspace=1.5)
    plt.show()