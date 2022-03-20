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
image_width = 320
image_height = 320
image_channels = 1

weights_output = r'C:\Users\rytis\Desktop\major_review_disertation/Drilling_output/'

training_data_dir = r'C:\Users\rytis\Desktop\major_review_disertation\dataForTraining_v3\dataForTraining_v3/'
testing_data_dir = r'C:\Users\rytis\Desktop\major_review_disertation\test_crop/'

# will be changed on each training
output_dir = ''

# train
train_images_dir = 'data/image/'
train_labels_dir = 'data/label/'
# test
test_images_dir = 'data/image/'
test_labels_dir = 'data/label/'

# batch size. How many samples you want to feed in one iteration?
batch_size = 8
# number_of_epoch. How many epochs you want to train?
number_of_epoch = 2
# initial learning rate
initial_lr = 0.001
# After how many epochs you want to reduce learning rate by half?
lr_scheduling_epochs = 4


class CustomSaver(tf.keras.callbacks.Callback):
    def __init__(self):
        self.best_val_score = 0.0
        self.iteration_counter = 0

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

    def on_batch_end(self, batch, logs=None):
        self.iteration_counter = self.iteration_counter + 1
        if self.iteration_counter % 1000 == 0:
            self.model.save(output_dir + f'_{self.iteration_counter}.hdf5')


# This function keeps the learning rate at 0.001 for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch):
    step = epoch // lr_scheduling_epochs
    lr = initial_lr / 2 ** step
    print('Epoch: ' + str(epoch) + ', learning rate = ' + str(lr))
    return lr


def train():
    global output_dir
    global weights_output
    architectures = ['UNet4',
                     'UNet4_SE',
                     'UNet4_coord',
                     'UNet4_coord_SE',
                     'UNet4_res_aspp',
                     'UNet4_res_aspp_SE',
                     'UNet4_res_aspp_coord',
                     'UNet4_res_aspp_coord_SE']

    for configuration_name in architectures:
        tf.keras.backend.clear_session()
        output_dir = f'{weights_output}/{configuration_name}//'

        train_images = f'{training_data_dir}/Image_rois/'
        train_labels = f'{training_data_dir}/Label_rois/'
        test_images = f'{testing_data_dir}/image/'
        test_labels = f'{testing_data_dir}/label/'

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

        # Define data generator that will take images from directory
        train_data_generator = data_generator(batch_size,
                                              image_folder=train_images,
                                              label_folder=train_labels,
                                              target_size=(image_width, image_height),
                                              image_color_mode='grayscale')

        test_data_generator = data_generator(batch_size,
                                             image_folder=test_images,
                                             label_folder=test_labels,
                                             target_size=(image_width, image_height),
                                             image_color_mode='grayscale')
        # create weights output directory
        if not os.path.exists(output_dir):
            print('Output directory doesnt exist!\n')
            print('It will be created!\n')
            os.makedirs(output_dir)

        tf.keras.backend.clear_session()
        if configuration_name == 'UNet4':
            print('*' * 50)
            print('UNet4')
            model = unet_autoencoder(filters_in_input=16,
                                     input_size=(320, 320, 1),
                                     learning_rate=1e-3,
                                     use_se=False,
                                     use_aspp=False,
                                     use_coord_conv=False,
                                     use_residual_connections=False,
                                     downscale_times=4,
                                     leaky_relu_alpha=0.1)
        elif configuration_name == 'UNet4_SE':
            print('*' * 50)
            print('UNet4_SE')
            model = unet_autoencoder(filters_in_input=16,
                                     input_size=(320, 320, 1),
                                     learning_rate=1e-3,
                                     use_se=True,
                                     use_aspp=False,
                                     use_coord_conv=False,
                                     use_residual_connections=False,
                                     downscale_times=4,
                                     leaky_relu_alpha=0.1)
        elif configuration_name == 'UNet4_coord':
            print('*' * 50)
            print('UNet4_coord')
            model = unet_autoencoder(filters_in_input=16,
                                     input_size=(320, 320, 1),
                                     learning_rate=1e-3,
                                     use_se=False,
                                     use_aspp=False,
                                     use_coord_conv=True,
                                     use_residual_connections=False,
                                     downscale_times=4,
                                     leaky_relu_alpha=0.1)
        elif configuration_name == 'UNet4_coord_SE':
            print('*' * 50)
            print('UNet4_coord_SE')
            model = unet_autoencoder(filters_in_input=16,
                                     input_size=(320, 320, 1),
                                     learning_rate=1e-3,
                                     use_se=True,
                                     use_aspp=False,
                                     use_coord_conv=True,
                                     use_residual_connections=False,
                                     downscale_times=4,
                                     leaky_relu_alpha=0.1)
        elif configuration_name == 'UNet4_res_aspp':
            print('*' * 50)
            print('UNet4_res_aspp')
            model = unet_autoencoder(filters_in_input=16,
                                     input_size=(320, 320, 1),
                                     learning_rate=1e-3,
                                     use_se=False,
                                     use_aspp=True,
                                     use_coord_conv=False,
                                     use_residual_connections=True,
                                     downscale_times=4,
                                     leaky_relu_alpha=0.1)
        elif configuration_name == 'UNet4_res_aspp_SE':
            print('*' * 50)
            print('UNet4_res_aspp_SE')
            model = unet_autoencoder(filters_in_input=16,
                                     input_size=(320, 320, 1),
                                     learning_rate=1e-3,
                                     use_se=True,
                                     use_aspp=True,
                                     use_coord_conv=False,
                                     use_residual_connections=True,
                                     downscale_times=4,
                                     leaky_relu_alpha=0.1)
        elif configuration_name == 'UNet4_res_aspp_coord':
            print('*' * 50)
            print('UNet4_res_aspp_coord')
            model = unet_autoencoder(filters_in_input=16,
                                     input_size=(320, 320, 1),
                                     learning_rate=1e-3,
                                     use_se=False,
                                     use_aspp=True,
                                     use_coord_conv=True,
                                     use_residual_connections=True,
                                     downscale_times=4,
                                     leaky_relu_alpha=0.1)
        elif configuration_name == 'UNet4_res_aspp_coord_SE':
            print('*' * 50)
            print('UNet4_res_aspp_coord_SE')
            model = unet_autoencoder(filters_in_input=16,
                                     input_size=(320, 320, 1),
                                     learning_rate=1e-3,
                                     use_se=True,
                                     use_aspp=True,
                                     use_coord_conv=True,
                                     use_residual_connections=True,
                                     downscale_times=4,
                                     leaky_relu_alpha=0.1)

        # # Define template of each epoch weight name. They will be save in separate files
        # weights_name = weights_output_dir + weights_output_name + "-{epoch:03d}-{loss:.4f}.hdf5"
        # Custom saving for the best-performing weights
        saver = CustomSaver()
        # Learning rate scheduler
        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

        model.fit(train_data_generator,
                  steps_per_epoch=number_of_train_iterations,
                  epochs=number_of_epoch,
                  validation_data=test_data_generator,
                  validation_steps=number_of_test_iterations,
                  callbacks=[saver, learning_rate_scheduler],
                  shuffle=True)

if __name__ == "__main__":
    train()
