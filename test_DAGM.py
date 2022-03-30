import glob
import os
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from models.autoencoder import unet_autoencoder
from models.losses import Loss
from models.data_loader import data_generator
from utilities import gather_image_from_dir

# Data
image_width = 512
image_height = 512
image_channels = 1

weights_output = r'D:\DAGM_output/'

data_dir = r'C:\Users\rytis\Desktop\major_review_disertation\DAGM/'

# will be changed on each training
output_dir = ''

# train
train_images_dir = 'data/image/'
train_labels_dir = 'data/label/'
# test
test_images_dir = 'data/image/'
test_labels_dir = 'data/label/'

# batch size. How many samples you want to feed in one iteration?
batch_size = 1
# number_of_epoch. How many epochs you want to train?
number_of_epoch = 1
# initial learning rate
initial_lr = 0.001


class CustomSaver(tf.keras.callbacks.Callback):
    def __init__(self):
        self.best_val_score = 0.0

    def on_epoch_end(self, epoch, logs=None):
        # also save if validation error is smallest
        if 'val_dice_eval' in logs.keys():
            val_score = logs['val_dice_eval']
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                print('New best weights found!')
                self.model.save(output_dir + 'best_weights.hdf5')
        else:
            print('Key val_dice_eval does not exist!')
        val_score = logs['val_dice_eval']
        self.model.save(output_dir + f'_{val_score}.hdf5')


def train():
    global output_dir
    global weights_output
    kernels = [8, 16, 32]
    layers = [2, 3, 4, 5]
    # gather all the classes
    classes_folders = glob.glob(data_dir + '*/')
    for classes_folder in classes_folders:
        for kernel in kernels:
            for layer in layers:
                tf.keras.backend.clear_session()
                folder_path = os.path.basename(classes_folder)
                output_dir = f'{weights_output}/{folder_path}/k{kernel}l{layer}/'

                # get weights from directory, it should be one file


                train_images = f'{classes_folder}/Train/image/'
                train_labels = f'{classes_folder}/Train/label/'
                test_images = f'{classes_folder}/Test/image/'
                test_labels = f'{classes_folder}/Test/label/'

                # check how many train and test samples are in the directories
                train_images_count = len(gather_image_from_dir(train_images))
                train_labels_count = len(gather_image_from_dir(train_labels))
                train_samples_count = min(train_images_count, train_labels_count)
                print('Training samples: ' + str(train_samples_count))

                test_images_count = len(gather_image_from_dir(test_images))
                test_labels_count = len(gather_image_from_dir(test_labels))
                test_samples_count = min(test_images_count, test_labels_count)
                print('Testing samples: ' + str(test_samples_count))

                # how many iterations in one epoch? Should cover whole dataset. Divide number of data samples from batch size
                number_of_train_iterations = train_samples_count // batch_size
                number_of_test_iterations = test_samples_count // batch_size

                # Define model
                model = unet_autoencoder(filters_in_input=kernel,
                                         input_size=(image_width, image_width, image_channels),
                                         loss_function=Loss.CROSSENTROPY50DICE50,
                                         downscale_times=layer,
                                         learning_rate=1e-3,
                                         use_se=False,
                                         use_aspp=False,
                                         use_coord_conv=False,
                                         use_residual_connections=False,
                                         leaky_relu_alpha=0.0)

                model.summary()

                # Define data generator that will take images from directory
                train_data_generator = data_generator(batch_size,
                                                      image_folder=train_images_dir,
                                                      label_folder=train_labels_dir,
                                                      target_size=(image_width, image_height),
                                                      image_color_mode='grayscale')

                test_data_generator = data_generator(batch_size,
                                                     image_folder=test_images_dir,
                                                     label_folder=test_labels_dir,
                                                     target_size=(image_width, image_height),
                                                     image_color_mode='grayscale')
                # create weights output directory
                if not os.path.exists(output_dir):
                    print('Output directory doesnt exist!\n')
                    print('It will be created!\n')
                    os.makedirs(output_dir)

                # # Define template of each epoch weight name. They will be save in separate files
                # weights_name = weights_output_dir + weights_output_name + "-{epoch:03d}-{loss:.4f}.hdf5"
                # Custom saving for the best-performing weights
                saver = CustomSaver()
                # # Make checkpoint for saving each
                # model_checkpoint = tf.keras.callbacks.ModelCheckpoint(weights_name, monitor='loss', verbose=1, save_best_only=False,
                #                                                       save_weights_only=False)
                model.fit(train_data_generator,
                          steps_per_epoch=number_of_train_iterations,
                          epochs=number_of_epoch,
                          validation_data=test_data_generator,
                          validation_steps=number_of_test_iterations,
                          callbacks=[saver],
                          shuffle=True)


if __name__ == "__main__":
    train()
