import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
    image_plot.title.set_fontsize(25)

    for tick in image_plot.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    for tick in image_plot.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    im = plt.imshow(image, cmap=colorspace, vmin=min, vmax=max)
    divider = make_axes_locatable(image_plot)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(17)


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
    fig = plt.figure(figsize=(2*6.6, 2*4.8))
    norm_image = image / 255.
    colormap = 'gray_r' if reverse_colormap else 'gray'
    vmin = 0.0
    vmax = 1.0
    add_subplot(fig, 1, 1, 1, name, norm_image, colormap, vmin, vmax)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)




def main():
    # image_400_exposure = cv2.imread(r'C:\src\personal\special_training_routines\special_training_routines/diagrams/exposure_400.png', cv2.IMREAD_GRAYSCALE)
    # image_200_exposure = cv2.imread(r'C:\src\personal\special_training_routines\special_training_routines/diagrams/exposure_200.png', cv2.IMREAD_GRAYSCALE)
    #
    # make_single_graph_grayscale('Exposure 200ns', image_200_exposure, 'exp200.png', False)
    # make_single_graph_grayscale('Exposure 400ns', image_400_exposure, 'exp400.png', False)

    different_depth = cv2.imread(
        r'C:\src\personal\special_training_routines\special_training_routines/diagrams/different_depth_drilling.png',
        cv2.IMREAD_GRAYSCALE)
    not_even_light_damaged_conveyor_belt = cv2.imread(
        r'C:\src\personal\special_training_routines\special_training_routines/diagrams/damaged_belt_not_even_light.png',
        cv2.IMREAD_GRAYSCALE)
    defective_lamination = cv2.imread(
        r'C:\src\personal\special_training_routines\special_training_routines/diagrams/defective_lamination.png',
        cv2.IMREAD_GRAYSCALE)

    #make_single_graph_grayscale('Different depth drilling', different_depth, 'Different depth drilling.png', False)
    #make_single_graph_grayscale('Damaged belt / not even light', not_even_light_damaged_conveyor_belt, 'Damaged belt.png', False)
    make_single_graph_grayscale('Defective lamination', defective_lamination, 'Defective lamination.png', False)


if __name__ == '__main__':
    main()
