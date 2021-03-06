import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.lines as lines
import seaborn as sns


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
    image_plot.title.set_fontsize(17)

    for tick in image_plot.xaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    for tick in image_plot.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)

    im = plt.imshow(image, cmap=colorspace, vmin=min, vmax=max)
    divider = make_axes_locatable(image_plot)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(14)


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
    fig = plt.figure(figsize=(6.6, 4.8))
    norm_image = image
    colormap = 'gray_r' if reverse_colormap else 'gray'
    vmin = 0.0
    vmax = 255.0
    add_subplot(fig, 1, 1, 1, name, norm_image, colormap, vmin, vmax)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_histogram(name, image, vertical_line_x, save_path):
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    plt.title(name, fontsize=20)
    plt.hist(image.ravel(), 256, [0, 256])
    ax.set_xlabel('Pixel Intensity', fontsize=16)
    ax.set_ylabel('Pixel Count', fontsize=16)
    plt.grid(color='#d68b5c', linestyle='--', linewidth=0.5, alpha=0.5)
    X = [vertical_line_x, vertical_line_x]
    Y = [0, 2400]
    ax.plot(X, Y, color='red', linewidth=2)
    # plt.tight_layout()
    plt.ylim([0, 2400])

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)

    plt.savefig(save_path, bbox_inches='tight')
    # plt.close(fig)
    # sns.displot(hist, len(hist), [0,256])
    # plt.show()


def get_architecture_name(path):
    architecture_names = ['UNet4_aspp_res_aspp_coord_SE',
                          'UNet4_res_aspp_SE',
                          'UNet4_res_aspp_coord',
                          'UNet4_res_aspp',
                          'UNet4_coord_SE',
                          'UNet4_SE',
                          'UNet4_coord',
                          'UNet4']
    architecture_names_ = ['UNet4+res+aspp+coord+SE',
                           'UNet4+res+aspp+SE',
                           'UNet4+res+aspp+coord',
                           'UNet4+res+aspp',
                           'UNet4+coord+SE',
                           'UNet4+SE',
                           'UNet4+coord',
                           'UNet4']
    for i, architecture_name in enumerate(architecture_names):
        if architecture_name in path:
            return architecture_names_[i]


def form_output_diagram(image_path, label_path, predictions_paths, output_dir):
    image = cv2.imread(image_path)
    label = cv2.imread(label_path)
    predictions = []
    for predictions_path in predictions_paths:
        predictions.append(cv2.imread(predictions_path))
    # invert label and predictions and draw boarders
    label = invert_image(label)
    label = draw_image_boarders(label)
    for i in range(len(predictions)):
        predictions[i] = invert_image(predictions[i])
        predictions[i] = draw_image_boarders(predictions[i])
    image = draw_image_boarders(image)
    #
    image_name = get_image_name(image_path)
    output_dir += image_name
    output_dir += '/'
    make_output_directory(output_dir)
    make_single_graph_grayscale('Image', image, output_dir + image_name + '_image.jpg', False)
    label_name = get_image_name(label_path)
    make_single_graph_grayscale('Label', label, output_dir + label_name + '.jpg')
    for i, prediction in enumerate(predictions):
        prediction_name = get_image_name(predictions_paths[i])
        diagram_name = get_architecture_name(predictions_paths[i])
        make_single_graph_grayscale(diagram_name, prediction, output_dir + prediction_name + '.jpg')


def form_conventional_output_diagram(image_path, label_path, output_dir):
    image = cv2.imread(image_path)
    label = cv2.imread(label_path)
    # invert label and predictions and draw boarders
    label = invert_image(label)
    label = draw_image_boarders(label)
    image_b = draw_image_boarders(image)
    #
    image_name = get_image_name(image_path)
    output_dir += image_name
    output_dir += '/'
    make_output_directory(output_dir)
    make_single_graph_grayscale('Image', image_b, output_dir + image_name + '_image.jpg', False)
    label_name = get_image_name(label_path)
    make_single_graph_grayscale('Label', label, output_dir + label_name + '.jpg')
    # use conventional methods and make diagrams
    # Sobel x + y
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(sobel_x)
    abs_grad_y = cv2.convertScaleAbs(sobel_y)
    sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    sobel = invert_image(sobel)
    make_single_graph_grayscale('Sobel [X+Y]', sobel, output_dir + label_name + '_sobel.jpg')
    # laplace
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = invert_image(laplacian)
    # laplacian = np.uint8(laplacian)
    # laplacian = cv2.convertScaleAbs(laplacian)
    make_single_graph_grayscale('Laplacian', laplacian, output_dir + label_name + '_laplacian.jpg')
    # canny edge
    canny = cv2.Canny(image, 100, 200)
    # canny = invert_image(canny)
    make_single_graph_grayscale('Canny edges', canny, output_dir + label_name + '_canny.jpg', False)


def main():
    images_path = r'C:\src\personal\special_training_routines\special_training_routines\diagrams/9[800]_90_2_0_29.jpg'
    image = cv2.imread(images_path, cv2.IMREAD_GRAYSCALE)
    # save classification
    mask_histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    # check if some specific pixel value has bigger pixel count than value 0 (background)
    # mask_histogram = np.squeeze(mask_histogram)

    otsu_value, otsu_threshold = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    print(otsu_value)
    otsu_threshold = 255 - otsu_threshold

    plot_histogram('Histogram with indicated threshold', image, otsu_value, 'hist.png')
    cv2.imshow('otsu', otsu_threshold)

    make_single_graph_grayscale('Image', image, 'image.png', False)
    make_single_graph_grayscale('Binary image (Otsu threshold)', otsu_threshold, 'otsu.png', True)

    cv2.waitKey(1)
    g = 2


if __name__ == '__main__':
    main()
