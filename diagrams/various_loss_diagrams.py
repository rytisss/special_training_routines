import os
import glob
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def parse_label(label):
    if 'l4k32AutoEncoder4_5x5_CROSSENTROPY25DICE75_0_0.001_' in label:
        return r'$\mathit{L}_{\mathit{CE}}^{\mathbf{25\%}}$' + '+' + r'$\mathit{L}_{\mathit{D}}^{\mathbf{75\%}}$'
    if 'l4k32AutoEncoder4_5x5_CROSSENTROPY50DICE50_0_0.001_' in label:
        return r'$\mathit{L}_{\mathit{CE}}^{\mathbf{50\%}}$' + '+' + r'$\mathit{L}_{\mathit{D}}^{\mathbf{50\%}}$'
    if 'l4k32AutoEncoder4_5x5_CROSSENTROPY75DICE25_0_0.001_' in label:
        return r'$\mathit{L}_{\mathit{CE}}^{\mathbf{75\%}}$' + '+' + r'$\mathit{L}_{\mathit{D}}^{\mathbf{25\%}}$'
    if 'l4k32AutoEncoder4_5x5_CROSSENTROPY_0_0.001_' in label:
        return r'$\mathit{L}_{\mathit{CE}}$'
    if 'l4k32AutoEncoder4_5x5_DICE_0_0.001_' in label:
        return r'$\mathit{L}_{\mathit{D}}$'
    if 'l4k32AutoEncoder4_5x5_SURFACEnDICE_0_0.001_' in label:
        return r'$\mathit{L}_{\mathit{DB}}$'
    if 'l4k32AutoEncoder4_5x5_WEIGHTED60CROSSENTROPY_0_0.001_' in label:
        return r'$\mathit{L}_{\mathit{W60CE}}$'
    if 'l4k32AutoEncoder4_5x5_WEIGHTED70CROSSENTROPY_0_0.001_' in label:
        return r'$\mathit{L}_{\mathit{W70CE}}$'
    if 'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY25DICE75_0_0.001_' in label:
        return r'$\mathit{L}_{\mathit{WCE}}^{\mathbf{25\%}}$' + '+' + r'$\mathit{L}_{\mathit{D}}^{\mathbf{75\%}}$'
    if 'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY50DICE50_0_0.001_' in label:
        return r'$\mathit{L}_{\mathit{WCE}}^{\mathbf{50\%}}$' + '+' + r'$\mathit{L}_{\mathit{D}}^{\mathbf{50\%}}$'
    if 'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY75DICE25_0_0.001_' in label:
        return r'$\mathit{L}_{\mathit{WCE}}^{\mathbf{75\%}}$' + '+' + r'$\mathit{L}_{\mathit{D}}^{\mathbf{25\%}}$'
    if 'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY_0_0.001_' in label:
        return r'$\mathit{L}_{\mathit{WCE}}$'


def makeGraphAll(x, y, x_all, y_all, name, output='', plot_width=8, plot_height=5):
    # just print all architecture names
    for architecture_name in x:
        print("'" + architecture_name + "'")
    print('Making diagram..')
    # translate labels
    x_trans = []
    for i in range(0, len(x)):
        x_trans.append(parse_label(x[i]))

    y_pos = np.arange(len(y))
    # separate list into 2 lists: random data feed results and sequence data feed results
    x1 = []
    x2 = []

    y1 = []
    y2 = []
    x_ = []
    for i in range(0, len(x_trans)):
        if i % 2 == 0:
            x2.append(x_trans[i])
            y2.append(y[i])
        else:
            x1.append(x_trans[i])
            y1.append(y[i])
        x_.append(x_trans[i])

    # plt.figure(1, figsize=(12,16))
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    # Turn on the minor TICKS, which are required for the minor GRID
    ax.minorticks_on()
    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.4)
    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='red', alpha=0.2)
    xlab = np.arange(len(x_))  # the label locations
    width = 0.6  # the width of the bars
    rects1 = ax.bar(xlab, y, width, label='Average', color='#59a4f0', edgecolor='#091229')
    # rects2 = ax.bar(xlab + width / 2, y1, width, label='Sequence', color = '#ff7700', edgecolor = '#091229')

    # first set
    ax.plot(xlab, y_all[0], color="#264653", marker='s', markerfacecolor="#264653", markeredgecolor='#1a2f38',
            label="Set_0", markeredgewidth=0.7, markersize=8)
    ax.plot(xlab, y_all[1], color="#2A9D8F", marker='o', markerfacecolor="#2A9D8F", markeredgecolor='#185951',
            label="Set_1", markeredgewidth=0.7, markersize=8)
    ax.plot(xlab, y_all[2], color="#E9C46A", marker='X', markerfacecolor="#E9C46A", markeredgecolor='#3d331a',
            label="Set_2", markeredgewidth=0.7, markersize=8)
    ax.plot(xlab, y_all[3], color="#F4A261", marker='D', markerfacecolor="#F4A261", markeredgecolor='#422c1a',
            label="Set_3", markeredgewidth=0.7, markersize=8)
    ax.plot(xlab, y_all[4], color="#E76F51", marker='^', markerfacecolor="#E76F51", markeredgecolor='#5c2b1f',
            label="Set_4", markeredgewidth=0.7, markersize=8)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Scores')
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(xlab)
    ax.set_xticklabels(x_, rotation='vertical')
    # ax.legend(loc='upper left')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('Dice Score')
    plt.xlabel('Loss function')
    plt.title(name)
    upper_limit1 = max(y1)
    upper_limit2 = max(y2)
    upper_limit = upper_limit1
    if upper_limit < upper_limit2:
        upper_limit = upper_limit2
    upper_limit += 0.0105
    upper_limit = 0.71
    plt.ylim((0.64, upper_limit))

    for rect, label in zip(rects1, y):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, 0.6415, str(format(label, '.4f')),
                ha='center', rotation='vertical', fontsize=10)

    """
    for rect, label in zip(rects2, y1):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height - 0.011, str(format(label, '.4f')),
                ha='center', rotation='vertical', fontsize=10)

        # Vertical alignment for positive values
        va = 'bottom'
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        space = 0.005
        # Create annotation
        ax.annotate(
            label,  # Use `label` as label
            (x_value, y_value),  # Place label at end of the bar
            xytext=(0, space),  # Vertically shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            ha='center',  # Horizontally center label
            va=va,
        )  # Vertically align label differently for
        # positive and negative values.
        """

    plt.tight_layout()
    # function to show the plot
    # plt.show()
    fig.savefig(output, dpi=400)
    plt.close()


def makeGraph(x, y, name, output='', plot_width=8, plot_height=5):
    # just print all architecture names
    for architecture_name in x:
        print("'" + architecture_name + "'")
    print('Making diagram..')
    # translate labels
    x_trans = []
    for i in range(0, len(x)):
        x_trans.append(parse_label(x[i]))

    y_pos = np.arange(len(y))
    # separate list into 2 lists: random data feed results and sequence data feed results
    x1 = []
    x2 = []

    y1 = []
    y2 = []
    x_ = []
    for i in range(0, len(x_trans)):
        if i % 2 == 0:
            x2.append(x_trans[i])
            y2.append(y[i])
        else:
            x1.append(x_trans[i])
            y1.append(y[i])
        x_.append(x_trans[i])

    # plt.figure(1, figsize=(12,16))
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    # Turn on the minor TICKS, which are required for the minor GRID
    ax.minorticks_on()
    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.4)
    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='red', alpha=0.2)
    xlab = np.arange(len(x_))  # the label locations
    width = 0.8  # the width of the bars
    rects1 = ax.bar(xlab, y, width, label='Average', color='#59a4f0', edgecolor='#091229')
    # rects2 = ax.bar(xlab + width / 2, y1, width, label='Sequence', color = '#ff7700', edgecolor = '#091229')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Scores')
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(xlab)
    ax.set_xticklabels(x_, rotation='vertical')
    ax.legend(loc='upper left')
    plt.ylabel('Dice Score')
    plt.xlabel('Loss function')
    plt.title(name)
    upper_limit1 = max(y1)
    upper_limit2 = max(y2)
    upper_limit = upper_limit1
    if upper_limit < upper_limit2:
        upper_limit = upper_limit2
    upper_limit += 0.0105
    upper_limit = 0.73
    plt.ylim((0.65, upper_limit))

    for rect, label in zip(rects1, y):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height - 0.011, str(format(label, '.4f')),
                ha='center', rotation='vertical', fontsize=10)

    """
    for rect, label in zip(rects2, y1):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height - 0.011, str(format(label, '.4f')),
                ha='center', rotation='vertical', fontsize=10)

        # Vertical alignment for positive values
        va = 'bottom'
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        space = 0.005
        # Create annotation
        ax.annotate(
            label,  # Use `label` as label
            (x_value, y_value),  # Place label at end of the bar
            xytext=(0, space),  # Vertically shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            ha='center',  # Horizontally center label
            va=va,
        )  # Vertically align label differently for
        # positive and negative values.
        """

    plt.tight_layout()
    # function to show the plot
    # plt.show()
    fig.savefig(output, dpi=400)
    plt.close()


def average_results(list_of_list):
    sum_list = []
    # take first list in list array size
    size = len(list_of_list[0])
    for i in range(0, size):
        sum_list.append(0)
    for list_i in list_of_list:
        for i in range(0, len(list_i)):
            sum_list[i] += list_i[i]
    number_of_lists = len(list_of_list)
    # average
    averages = []
    for i in range(0, len(sum_list)):
        averages.append(sum_list[i] / (float)(number_of_lists))
    return averages


def get_results_from_order(x, y, order):
    x_sort = []
    y_sort = []
    for i in range(0, len(order)):
        for j in range(0, len(x)):
            order_elem = order[i]
            x_elem = x[j]
            if order_elem in x_elem:
                x_sort.append(x[j])
                # x_sort.append(x[j+1])
                y_sort.append(y[j])
                # y_sort.append(y[j+1])
    return x_sort, y_sort


def get_single_loss_res_ordered(x, y):
    print('Ordering single loss funtion results')
    order = [
        'l4k32AutoEncoder4_5x5_CROSSENTROPY_0_0.001_',
        'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY_0_0.001_',
        'l4k32AutoEncoder4_5x5_WEIGHTED60CROSSENTROPY_0_0.001_',
        'l4k32AutoEncoder4_5x5_WEIGHTED70CROSSENTROPY_0_0.001_',
        'l4k32AutoEncoder4_5x5_DICE_0_0.001_'
    ]
    return get_results_from_order(x, y, order)


def get_combo_loss_res_ordered(x, y):
    print('Ordering combo loss funtion results')
    order = [
        'l4k32AutoEncoder4_5x5_CROSSENTROPY25DICE75_0_0.001_',
        'l4k32AutoEncoder4_5x5_CROSSENTROPY50DICE50_0_0.001_',
        'l4k32AutoEncoder4_5x5_CROSSENTROPY75DICE25_0_0.001_',
        'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY25DICE75_0_0.001_',
        'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY50DICE50_0_0.001_',
        'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY75DICE25_0_0.001_',
        'l4k32AutoEncoder4_5x5_SURFACEnDICE_0_0.001_'
    ]
    return get_results_from_order(x, y, order)


def get_all_results_in_order(x, y):
    print('Ordering all results')
    order = [
        'l4k32AutoEncoder4_5x5_CROSSENTROPY_0_0.001_',
        'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY_0_0.001_',
        'l4k32AutoEncoder4_5x5_WEIGHTED60CROSSENTROPY_0_0.001_',
        'l4k32AutoEncoder4_5x5_WEIGHTED70CROSSENTROPY_0_0.001_',
        'l4k32AutoEncoder4_5x5_DICE_0_0.001_',
        'l4k32AutoEncoder4_5x5_CROSSENTROPY25DICE75_0_0.001_',
        'l4k32AutoEncoder4_5x5_CROSSENTROPY50DICE50_0_0.001_',
        'l4k32AutoEncoder4_5x5_CROSSENTROPY75DICE25_0_0.001_',
        'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY25DICE75_0_0.001_',
        'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY50DICE50_0_0.001_',
        'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY75DICE25_0_0.001_',
        'l4k32AutoEncoder4_5x5_SURFACEnDICE_0_0.001_'
    ]
    return get_results_from_order(x, y, order)


def main():
    sets = ['Set_0', 'Set_1', 'Set_2', 'Set_3', 'Set_4']
    all_x = []
    all_y = []
    single_loss_fig_size = (4.5, 4.5)
    combo_loss_fig_size = (5.5, 5)
    all_loss_fig_size = (8, 5)
    every_single_x = []
    every_single_y = []
    every_combo_y = []
    every_combo_x = []
    every_all_x = []
    every_all_y = []
    # for set in sets:
    #     architecturesInputDir = 'D:/CracksTrainings/' + set + '/'
    #     # Get subdirectories of all architectures
    #     inputArchitecturesSubDirs = glob.glob(architecturesInputDir + '*/')
    #     x = []
    #     y = []
    #     for inputArchitectureSurDir in inputArchitecturesSubDirs:
    #         if 'SEQ' in inputArchitectureSurDir:
    #             continue  # skip
    #         txt_path = inputArchitectureSurDir + 'averageScore.txt'
    #         dir_name = os.path.basename(os.path.normpath(inputArchitectureSurDir))
    #         input_file = open(txt_path, 'r')
    #         currentBenchmark = Benchmark()
    #         count = 0
    #         for lineText in input_file:
    #             # skip first line
    #             if count == 0:
    #                 count = count + 1
    #                 continue
    #             currentBenchmark.parseDataLine(lineText, count - 1)
    #             count += 1
    #         x.append(dir_name)
    #         score, epoch = currentBenchmark.GetBestDice()
    #         y.append(score)
    #         input_file.close()
    #     output = 'benchmarks/'
    #     os.makedirs(output, exists=True)
    #     x_single, y_single = get_single_loss_res_ordered(x, y)
    #     x_combo, y_combo = get_combo_loss_res_ordered(x, y)
    #     x_all, y_all = get_all_results_in_order(x, y)
    #
    #     # collect sorted data
    #     every_single_x.append(x_single)
    #     every_single_y.append(y_single)
    #     every_combo_x.append(x_combo)
    #     every_combo_y.append(y_combo)
    #     every_all_x.append(x_all)
    #     every_all_y.append(y_all)
    #
    #     """
    #     makeGraph(x_single, y_single, 'Best Dice Scores of ' + set, output + set + '_single.png', single_loss_fig_size[0], single_loss_fig_size[1])
    #     makeGraph(x_combo, y_combo, 'Best Dice Scores of ' + set, output + set + '_combo.png', combo_loss_fig_size[0], combo_loss_fig_size[1])
    #     makeGraph(x_all, y_all, 'Best Dice Scores of ' + set, output + set + '_all.png', all_loss_fig_size[0], all_loss_fig_size[1])
    #     """
    #     all_x.append(x)
    #     all_y.append(y)
    output = 'various_loss/'
    os.makedirs(output, exist_ok=True)

    # CROSSENTROPY25DICE75
    # CROSSENTROPY50DICE50
    # CROSSENTROPY75DICE25
    # CROSSENTROPY
    # DICE
    # SURFACEnDICE
    # WEIGHTED60CROSSENTROPY
    # WEIGHTED70CROSSENTROPY
    # WEIGHTEDCROSSENTROPY25DICE75
    # WEIGHTEDCROSSENTROPY50DICE50
    # WEIGHTEDCROSSENTROPY75DICE25
    # WEIGHTEDCROSSENTROPY

    set_0_dice = np.array([0.6937,
                           0.6913,
                           0.6895,
                           0.6699,
                           0.6919,
                           0.692,
                           0.6931,
                           0.6801,
                           0.6976,
                           0.6951,
                           0.6912,
                           0.6926])

    set_1_dice = np.array([0.6842,
                           0.6903,
                           0.679,
                           0.6608,
                           0.6814,
                           0.6841,
                           0.6803,
                           0.6774,
                           0.6805,
                           0.6798,
                           0.6818,
                           0.6849
                           ])

    set_2_dice = np.array([0.6826,
                           0.6791,
                           0.6752,
                           0.6759,
                           0.6835,
                           0.6772,
                           0.6799,
                           0.6752,
                           0.6848,
                           0.6854,
                           0.6825,
                           0.6788
                           ])

    set_3_dice = np.array([0.7056,
                           0.7066,
                           0.707,
                           0.6867,
                           0.7063,
                           0.7043,
                           0.7083,
                           0.6975,
                           0.7014,
                           0.7048,
                           0.6978,
                           0.7049
                           ])

    set_4_dice = np.array([0.6856,
                           0.6898,
                           0.6923,
                           0.6744,
                           0.691,
                           0.6985,
                           0.6882,
                           0.6856,
                           0.6861,
                           0.6878,
                           0.6955,
                           0.69
                           ])

    average_dice = np.array([0.6903,
                             0.6914,
                             0.6886,
                             0.6735,
                             0.6908,
                             0.6912,
                             0.6900,
                             0.6832,
                             0.6901,
                             0.6906,
                             0.6898,
                             0.6902])

    single_order = [
        'l4k32AutoEncoder4_5x5_CROSSENTROPY_0_0.001_',
        'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY_0_0.001_',
        'l4k32AutoEncoder4_5x5_WEIGHTED60CROSSENTROPY_0_0.001_',
        'l4k32AutoEncoder4_5x5_WEIGHTED70CROSSENTROPY_0_0.001_',
        'l4k32AutoEncoder4_5x5_DICE_0_0.001_'
    ]

    combo_order = [
        'l4k32AutoEncoder4_5x5_CROSSENTROPY25DICE75_0_0.001_',
        'l4k32AutoEncoder4_5x5_CROSSENTROPY50DICE50_0_0.001_',
        'l4k32AutoEncoder4_5x5_CROSSENTROPY75DICE25_0_0.001_',
        'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY25DICE75_0_0.001_',
        'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY50DICE50_0_0.001_',
        'l4k32AutoEncoder4_5x5_WEIGHTEDCROSSENTROPY75DICE25_0_0.001_',
        'l4k32AutoEncoder4_5x5_SURFACEnDICE_0_0.001_'
    ]

    single_indices = [3, 11, 6, 7, 4]
    combo_indices = [0, 1, 2, 8, 9, 10, 5]
    x_single = average_dice[single_indices]
    y_single = single_order
    every_single_x.append(set_0_dice[single_indices])
    every_single_x.append(set_1_dice[single_indices])
    every_single_x.append(set_2_dice[single_indices])
    every_single_x.append(set_3_dice[single_indices])
    every_single_x.append(set_4_dice[single_indices])
    every_single_y.append(single_order)
    every_single_y.append(single_order)
    every_single_y.append(single_order)
    every_single_y.append(single_order)
    every_single_y.append(single_order)

    x_combo = average_dice[combo_indices]
    y_combo = combo_order
    every_combo_x.append(set_0_dice[combo_indices])
    every_combo_x.append(set_1_dice[combo_indices])
    every_combo_x.append(set_2_dice[combo_indices])
    every_combo_x.append(set_3_dice[combo_indices])
    every_combo_x.append(set_4_dice[combo_indices])
    every_combo_y.append(combo_order)
    every_combo_y.append(combo_order)
    every_combo_y.append(combo_order)
    every_combo_y.append(combo_order)
    every_combo_y.append(combo_order)

    # average_y = average_results(all_y)
    # x_single, y_single = get_single_loss_res_ordered(x, average_y)
    # x_combo, y_combo = get_combo_loss_res_ordered(x, average_y)
    # x_all, y_all = get_all_results_in_order(x, average_y)
    # makeGraph(x_single, y_single, 'Best Dice Scores (Average)', output + '_single.png', single_loss_fig_size[0],
    #           single_loss_fig_size[1])
    # makeGraph(x_combo, y_combo, 'Best Dice Scores (Average)', output + '_combo.png', combo_loss_fig_size[0],
    #           combo_loss_fig_size[1])
    # makeGraph(x_all, y_all, 'Best Dice Scores (Average)', output + '_all.png', all_loss_fig_size[0],
    #           all_loss_fig_size[1])

    makeGraphAll(y_single, x_single, every_single_y, every_single_x, 'Best Dice Scores (Single Loss Function)',
                 output + '_singleEvery.png', combo_loss_fig_size[0],
                 combo_loss_fig_size[1])
    makeGraphAll(y_combo, x_combo, every_combo_y, every_combo_x, 'Best Dice Scores (Combo Loss Function)',
                 output + '_comboEvery.png', combo_loss_fig_size[0],
                 combo_loss_fig_size[1])
    # makeGraphAll(x_all, y_all, every_all_x, every_all_y, 'Best Dice Scores (Average)', output + '_allEvery.png',
    #              all_loss_fig_size[0],
    #              all_loss_fig_size[1])


if __name__ == "__main__":
    main()
