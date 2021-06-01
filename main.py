from random import random

from cv2 import cv2
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
# import cv2
import scipy.io
import tensorflow as tf
from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow.keras.datasets as tfds
import os
import sys
import glob
import urllib
import urllib.request
import tarfile
import numpy as np
from scipy.io import loadmat
from urllib.request import urlopen
from shutil import copyfileobj
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.python.keras.callbacks import Callback
from tensorboard.plugins.hparams import api as hp
from tensorflow.python.keras.layers import Dropout

labels_names = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold',
                'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon',
                "colt's foot",
                'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily',
                'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger',
                'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke',
                'sweet william',
                'carnation', 'garden phlox', 'love in the mist', 'mexican aster', 'alpine sea holly',
                'ruby-lipped cattleya',
                'cape flower', 'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil',
                'sword lily',
                'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'oxeye daisy',
                'common dandelion',
                'petunia', 'wild pansy', 'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura',
                'geranium',
                'orange dahlia', 'pink-yellow dahlia?', 'cautleya spicata', 'japanese anemone', 'black-eyed susan',
                'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris', 'windflower',
                'tree poppy', 'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory',
                'passion flower',
                'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose',
                'tree mallow', 'magnolia', 'cyclamen ', 'watercress', 'canna lily', 'hippeastrum ', 'bee balm',
                'ball moss',
                'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower',
                'trumpet creeper', 'blackberry lily']


def download_file(url, dest=None):
    if not dest:
        dest = 'data/' + url.split('/')[-1]
    with urlopen(url) as in_stream, open(dest, 'wb') as out_file:
        copyfileobj(in_stream, out_file)


def download_data():
    # Download the Oxford102 dataset into the current directory
    if not os.path.exists('data'):
        os.mkdir('data')

        print("Downloading images...")
        download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz')
        tarfile.open("data/102flowers.tgz").extractall(path='data/')

        print("Downloading image labels...")
        download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat')

        print("Downloading train/test/valid splits...")
        download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat')


def load_data(random_split=True):
    # Read .mat file containing image labels.
    image_labels = loadmat('data/imagelabels.mat')['labels'][0]
    # Subtract one to get 0-based labels
    image_labels -= 1

    all_files = sorted(glob.glob('data/jpg/*.jpg'))
    # fix path's backslashes
    for ind, file in enumerate(all_files):
        all_files[ind] = file.replace('\\', '/')
    if random_split:
        # split into train/test
        X_train, X_test, y_train, y_test = train_test_split(all_files, image_labels, test_size=0.25, random_state=SEED)
        # split into train/validation
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3333, random_state=SEED)

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_val, y_val = np.array(X_val), np.array(y_val)
        X_test, y_test = np.array(X_test), np.array(y_test)

    else:
        # Read .mat file containing training, testing, and validation sets.
        setid = loadmat('data/setid.mat')

        # The .mat file is 1-indexed, so we subtract one to get 0-based labels
        idx_train = setid['trnid'][0] - 1
        idx_test = setid['tstid'][0] - 1
        idx_valid = setid['valid'][0] - 1

        # zip together images paths and labels
        ziped_labels = zip(all_files, image_labels)
        labels = list(ziped_labels)
        labels = np.array(labels)

        # Images are ordered by species, so shuffle them
        np.random.seed(SEED)
        idx_train = idx_train[np.random.permutation(len(idx_train))]
        idx_test = idx_test[np.random.permutation(len(idx_test))]
        idx_valid = idx_valid[np.random.permutation(len(idx_valid))]

        # split into train, test and validation
        X_train, y_train = labels[idx_train, :][:, 0], labels[idx_train, :][:, 1]
        X_val, y_val = labels[idx_valid, :][:, 0], labels[idx_valid, :][:, 1]
        X_test, y_test = labels[idx_test, :][:, 0], labels[idx_test, :][:, 1]

    return X_train, y_train, X_val, y_val, X_test, y_test


def process_image(img):
    def image_center_crop(img):
        """
        https://github.com/antonio-f/Inception-V3/blob/master/TF2_InceptionV3/InceptionV3_fine_tuning.ipynb
        Makes a square center crop of an img, which is a [h, w, 3] numpy array.
        Returns [min(h, w), min(h, w), 3] output with same width and height.
        For cropping use numpy slicing.
        """
        h, w = img.shape[0], img.shape[1]
        m = min(h, w)
        cropped_img = img[(h - m) // 2:(h + m) // 2, (w - m) // 2:(w + m) // 2, :]

        return cropped_img

    image = np.squeeze(img)
    if crop:
        image = image_center_crop(image)
    if normalization:
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
    else:
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image


def generate_data(file_list, labels, batch_size):
    """Replaces Keras native ImageDataGenerator."""
    i = 0
    while True:
        image_batch = []
        labels_batch = []
        for b in range(batch_size):
            if i == len(file_list):
                i = 0
                # random.shuffle(file_list)
            labels_batch.append(labels[i])  # b
            image = cv2.imread(file_list[i])
            i += 1
            image = process_image(image)
            image_batch.append(image)

        image_batch = np.stack(image_batch, axis=0)
        batch_targets = tf.keras.utils.to_categorical(labels_batch, N_CLASSES)
        yield image_batch, batch_targets


def test_generate(file_list, label):
    images = []
    for file in file_list:
        image = cv2.imread(file)
        image = process_image(image)
        images.append(image)
    images = np.stack(images, axis=0)
    labels = tf.keras.utils.to_categorical(label, N_CLASSES)
    return images, labels


def display_flower(images_list, labels, flower_ind):
    import matplotlib.pyplot as plt
    image = cv2.imread(images_list[flower_ind])
    plt.imshow(image)
    plt.title(f"number: {labels[flower_ind]}, name: {labels_names[labels[flower_ind]]}")
    plt.show()


def get_vgg_adapted(hparams):
    # load model and specify a new input shape for images
    feature_extractor = VGG16(include_top=False, input_tensor=INPUT_SHAPE)
    # Freeze the Pre-Trained Model
    feature_extractor.trainable = False
    # add new classifier layers
    flat1 = Flatten()(feature_extractor.layers[-1].output)
    class1 = Dense(hparams[HP_NUM_UNITS], activation='relu')(flat1)
    class2 = Dropout(hparams[HP_DROPOUT])(class1)
    output = Dense(N_CLASSES, activation='softmax')(class2)
    # define new model
    model = Model(inputs=feature_extractor.inputs, outputs=output)
    # summarize
    model.summary()
    return model


def get_resnet_adapted(hparams):
    # load model and specify a new input shape for images
    feature_extractor = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    # Freeze the Pre-Trained Model
    feature_extractor.trainable = False
    # add new classifier layers
    flat1 = Flatten()(feature_extractor.layers[-1].output)
    class1 = Dense(hparams[HP_NUM_UNITS], activation='relu')(flat1)
    class2 = Dropout(hparams[HP_DROPOUT])(class1)
    output = Dense(N_CLASSES, activation='softmax')(class2)
    # define new model
    model = Model(inputs=feature_extractor.inputs, outputs=output)
    # summarize
    model.summary()
    return model


def get_mobilenet_v2_adapted(hparams):
    # Create a Feature Extractor
    URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    feature_extractor = hub.KerasLayer(URL, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    # Freeze the Pre-Trained Model
    feature_extractor.trainable = False
    # Attach a classification head
    model = tf.keras.Sequential([
        feature_extractor,
        Dense(hparams[HP_NUM_UNITS], activation='relu'),
        Dropout(hparams[HP_DROPOUT]),
        layers.Dense(N_CLASSES, activation='softmax')
    ])
    return model


# Implementation base on https://github.com/keras-team/keras/issues/2548
class TestCallback(Callback):
    def __init__(self, test_data):
        super().__init__()
        self.test_data = test_data
        self.history_test = {'test_accuracy': [], 'test_loss': []}

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        self.history_test['test_accuracy'].append(acc)
        self.history_test['test_loss'].append(loss)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


def plot(name_model, history, history_test, session_num):
    epochs_range = range(EPOCHS)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
    plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(epochs_range, history_test['test_accuracy'], label='Test Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['loss'], label='Training Loss')
    plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
    plt.plot(epochs_range, history_test['test_loss'], label='Test Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    if not os.path.exists('data/' + name_model):
        os.mkdir('data/' + name_model)
    plt.savefig('data/' + name_model + '/' + str(session_num) + '.png')

    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Let's go!")
    download_data()
    # we will crop and resize input images to IMG_SIZE x IMG_SIZE
    N_CLASSES = 102
    IMG_SIZE = 224
    BATCH_SIZE = 32
    # TODO: change for experiments
    SEED = 42
    EPOCHS = 100
    run_model = 'feature_vector'
    normalization = False
    crop = True

    HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([128, 256, 1024]))
    HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.0, 0.3]))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

    INPUT_SHAPE = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(random_split=True)

    X_test, y_test = test_generate(X_test, y_test)
    session_num = 0

    for num_units in HP_NUM_UNITS.domain.values:
        for dropout_rate in HP_DROPOUT.domain.values:
            for optimizer in HP_OPTIMIZER.domain.values:
                hparams = {
                    HP_NUM_UNITS: num_units,
                    HP_DROPOUT: dropout_rate,
                    HP_OPTIMIZER: optimizer,
                }
                if run_model == 'VGG16':
                    model = get_vgg_adapted(hparams)
                elif run_model == 'feature_vector':
                    model = get_mobilenet_v2_adapted(hparams)
                else:  # 'resnet'
                    model = get_resnet_adapted(hparams)
                print(f"-----------------------------------------------------------{run_model} "
                      f"--------------------------------------------------------------")
                model.compile(
                    optimizer=hparams[HP_OPTIMIZER],
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
                print(f"session number {session_num}")
                print({h.name: hparams[h] for h in hparams})
                # Stop training when there is no improvement in the validation loss for 5 consecutive epochs
                early_stopping = EarlyStopping(monitor='val_loss', patience=5)

                callable_test = TestCallback((X_test, y_test))
                history = model.fit(generate_data(X_train, y_train, BATCH_SIZE),
                                    steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                                    validation_data=generate_data(X_val, y_val, BATCH_SIZE),
                                    epochs=EPOCHS,
                                    validation_steps=X_val.shape[0] // BATCH_SIZE,
                                    callbacks=[early_stopping, callable_test])
                loss_test, acc_test = model.evaluate(X_test, y_test)
                str_loss_acc = "SN_{:.1f}_loss_{:.3f}_acc_{:.3f}".format(session_num, loss_test, acc_test)
                plot(run_model, history, callable_test.history_test, str_loss_acc)

                session_num += 1



# TODO:
#   1. pre processing
#       a. normalization - [0,1], [0, 255] - V
#       c. crop the center - V
#   2. add model - resnet - V
#   3. add layer to model - V
#   4. Tuning parameters - V
