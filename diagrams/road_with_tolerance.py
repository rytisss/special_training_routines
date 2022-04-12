import os
import glob
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def makeGraph(labels, y1, y2, y3, name, output='', plot_width=11, plot_height=6, ymin=0.0, ymax=1.0):
    # just print all architecture names
    print('Making diagram..')

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    rects1 = ax.bar(x - width, y1, width, label='0 px', edgecolor='#091229')
    rects2 = ax.bar(x, y2, width, label='2 px', edgecolor='#091229')
    rects3 = ax.bar(x + width, y3, width, label='5 px', edgecolor='#091229')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Dice score', fontsize=17)
    ax.set_xlabel('CNN Model', fontsize=17)
    ax.set_title(name, fontsize=17)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # ax.set_xticklabels(labels, rotation='vertical')
    legend = ax.legend(fontsize=14, title='Tolerance', title_fontsize=14, loc='upper center')
    #legend.set_title('Tolerance', title_fontsize=14)
    ax.minorticks_on()
    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.4)
    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='red', alpha=0.2)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            textOffset = (ymax - ymin) / 80.0
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, ymin + textOffset),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90, fontsize=19)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    plt.ylim((ymin, ymax))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=13)
    fig.tight_layout()

    # plt.show()

    # function to show the plot
    # plt.show()
    fig.savefig(output, dpi=400)
    plt.close()


def main():
    crackforest_labels = ['UNet', 'UNet + Res +\nASPP']
    crackforest_0_tolerance = [0.7015, 0.7114]
    crackforest_2_tolerance = [0.9486, 0.9570]
    crackforest_5_tolerance = [0.9694, 0.9729]

    # base directory for output
    output = ''

    makeGraph(crackforest_labels, crackforest_0_tolerance, crackforest_2_tolerance, crackforest_5_tolerance,
              'Models Dice Score with Tolerance\n on CrackForest Dataset',
              output + 'crackforest_tolerance.png', 7, 6, 0.4, 1.0)

    crack500_labels = ['UNet', 'UNet + Res +\nASPP_WF']
    crack500_0_tolerance = [0.6803, 0.6931]
    crack500_2_tolerance = [0.9070, 0.9161]
    crack500_5_tolerance = [0.9626, 0.9702]

    makeGraph(crack500_labels, crack500_0_tolerance, crack500_2_tolerance, crack500_5_tolerance,
              'Models Dice Score with Tolerance\n on Crack500 Dataset',
              output + 'crack500_tolerance.png', 7, 6, 0.4, 1.0)

    gaps384_labels = ['UNet', 'UNet + Res +\nASPP + AG']
    gaps384_0_tolerance = [0.5448, 0.5822]
    gaps384_2_tolerance = [0.8009, 0.8219]
    gaps384_5_tolerance = [0.8746, 0.8966]

    makeGraph(gaps384_labels, gaps384_0_tolerance, gaps384_2_tolerance, gaps384_5_tolerance,
              'Models Dice Score with Tolerance\n on GAPs384 Dataset',
              output + 'gaps384_tolerance.png', 7, 6, 0.4, 1.0)


if __name__ == "__main__":
    main()
