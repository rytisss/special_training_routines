import os
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from augmentation import Augmentation


def gather_image_from_dir(input_dir):
    image_extensions = ['*.jpg', '*.png', '*.bmp']
    image_list = []
    for image_extension in image_extensions:
        image_list.extend(glob.glob(input_dir + image_extension))
    image_list.sort()
    return image_list


def make_output_directory(output_dir):
    if not os.path.exists(output_dir):
        print('Making output directory: ' + output_dir)
        os.makedirs(output_dir)


def get_image_name(path):
    image_name_with_ext = path.rsplit('\\', 1)[1]
    image_name, image_extension = os.path.splitext(image_name_with_ext)
    return image_name


def find_image_with_name(name, image_paths):
    paths = []
    for image_path in image_paths:
        if name in image_path:
            paths.append(image_path)
    return paths


def draw_image_boarders(image):
    width, height = image.shape[:2]
    # draw border
    border_color = 0
    # invert image
    cv2.rectangle(image, (0, 0), (height - 1, width - 1), border_color, 1)
    return image


def invert_image(image):
    return abs(255 - image)


def add_subplot(fig, rows, cols, pos, name, image, colorspace, min, max):
    image_plot = fig.add_subplot(rows, cols, pos)
    image_plot.title.set_text(name)
    image_plot.title.set_fontsize(35)

    for tick in image_plot.xaxis.get_major_ticks():
        tick.label.set_fontsize(26)

    for tick in image_plot.yaxis.get_major_ticks():
        tick.label.set_fontsize(26)

    im = plt.imshow(image, cmap=colorspace, vmin=min, vmax=max)
    divider = make_axes_locatable(image_plot)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(26)


def make_single_graph(name, image, save_path):
    fig = plt.figure(figsize=(6.6, 4.8))
    norm_image = image / 255.
    colormap = 'viridis'
    vmin = 0.0
    vmax = 1.0
    add_subplot(fig, 1, 1, 1, name, norm_image, colormap, vmin, vmax)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def make_single_graph_grayscale(name, image, save_path, reverse_colormap=True):
    fig = plt.figure(figsize=(2 * 6.6, 2 * 4.8))
    norm_image = image / 255.
    colormap = 'gray_r' if reverse_colormap else 'gray'
    vmin = 0.0
    vmax = 1.0
    add_subplot(fig, 1, 1, 1, name, norm_image, colormap, vmin, vmax)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_histogram(name, image, save_path, max_y=None):
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    plt.title(name, fontsize=22)
    plt.hist(image.ravel(), 256, [0, 256])
    ax.set_xlabel('Pixel Intensity', fontsize=15)
    ax.set_ylabel('Pixel Count', fontsize=15)
    plt.grid(color='#d68b5c', linestyle='--', linewidth=0.5, alpha=0.5)
    # plt.tight_layout()
    if max_y:
        plt.ylim([0, max_y])

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    plt.savefig(save_path, bbox_inches='tight')
    # plt.close(fig)
    # sns.displot(hist, len(hist), [0,256])
    # plt.show()


def get_highest_image_value(image):
    max_value = 0
    for i in range(0, 256):
        occurencies = np.count_nonzero(image == i)
        if occurencies >= max_value:
            max_value = occurencies
    return max_value


def main():
    # image_400_exposure = cv2.imread(r'C:\src\personal\special_training_routines\special_training_routines/diagrams/exposure_400.png', cv2.IMREAD_GRAYSCALE)
    # image_200_exposure = cv2.imread(r'C:\src\personal\special_training_routines\special_training_routines/diagrams/exposure_200.png', cv2.IMREAD_GRAYSCALE)
    #
    # make_single_graph_grayscale('Exposure 200ns', image_200_exposure, 'exp200.png', False)
    # make_single_graph_grayscale('Exposure 400ns', image_400_exposure, 'exp400.png', False)
    image_paths = gather_image_from_dir(r'C:\src\personal\special_training_routines\special_training_routines\cases/')
    make_output_directory('cases_output/')
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        name = get_image_name(image_path)
        make_single_graph_grayscale('Image', image, 'cases_output/' + name + '.png', False)
        image_gamma_08 = Augmentation.Adjust_gamma(image, 0.8)
        image_gamma_12 = Augmentation.Adjust_gamma(image, 1.2)

        image_max_y = get_highest_image_value(image)
        image_gamma_08_max_y = get_highest_image_value(image_gamma_08)
        image_gamma_12_max_y = get_highest_image_value(image_gamma_12)

        max_y = max(max(image_max_y, image_gamma_08_max_y), image_gamma_12_max_y) + 100

        plot_histogram('Image', image, 'cases_output/' + name + '_hist.png', max_y)
        make_single_graph_grayscale('Gamma correction 1.2', image_gamma_12, 'cases_output/' + name + '_gamma_12.png',
                                    False)
        plot_histogram('Gamma correction 1.2', image_gamma_12, 'cases_output/' + name + '_gamma_12_hist.png', max_y)
        make_single_graph_grayscale('Gamma correction 0.8', image_gamma_08, 'cases_output/' + name + '_gamma_08.png',
                                    False)
        plot_histogram('Gamma correction 0.8', image_gamma_08, 'cases_output/' + name + '_gamma_08_hist.png', max_y)
        gaussian_noise = Augmentation.AddNoise(image)
        make_single_graph_grayscale('Gaussian noise', gaussian_noise, 'cases_output/' + name + '_gaussian_noise.png',
                                    False)
        image_gaussian = cv2.GaussianBlur(image, (3, 3), 0)
        make_single_graph_grayscale('Gaussian blur', image_gaussian, 'cases_output/' + name + '_blur.png',
                                    False)


if __name__ == '__main__':
    main()
