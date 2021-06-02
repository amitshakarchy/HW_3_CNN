from random import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
import cv2
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
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
import tensorflow_hub as hub

SEED = 42
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
    image = np.squeeze(img)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
    # TODO: crop the center of the image?
    # TODO: preprocess using: from keras.applications.vgg16 import preprocess_input (prepared_images = preprocess_input(images))
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


def display_flower(images_list, labels, flower_ind):
    import matplotlib.pyplot as plt
    image = cv2.imread(images_list[flower_ind])
    plt.imshow(image)
    plt.title(f"number: {labels[flower_ind]}, name: {labels_names[labels[flower_ind]]}")
    plt.show()


def get_vgg_adapted():
    # load model and specify a new input shape for images
    feature_extractor = VGG16(include_top=False, input_tensor=INPUT_SHAPE)
    # Freeze the Pre-Trained Model
    feature_extractor.trainable = False
    # add new classifier layers
    flat1 = Flatten()(feature_extractor.layers[-1].output)
    class1 = Dense(1024, activation='relu')(flat1)
    output = Dense(N_CLASSES, activation='softmax')(class1)
    # define new model
    model = Model(inputs=feature_extractor.inputs, outputs=output)
    # summarize
    model.summary()
    return model


def get_mobilenet_v2_adapted():
    # Create a Feature Extractor
    URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    feature_extractor = hub.KerasLayer(URL, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    # Freeze the Pre-Trained Model
    feature_extractor.trainable = False
    # Attach a classification head
    model = tf.keras.Sequential([
        feature_extractor,
        layers.Dense(N_CLASSES, activation='softmax')
    ])
    return model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Let's go!")
    # we will crop and resize input images to IMG_SIZE x IMG_SIZE
    N_CLASSES = 102
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 20
    INPUT_SHAPE = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(random_split=True)

    # TODO: Build and train your network.
    model = get_mobilenet_v2_adapted()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    # Stop training when there is no improvement in the validation loss for 5 consecutive epochs
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    history = model.fit(generate_data(X_train, y_train, BATCH_SIZE),
                        steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                        validation_data=generate_data(X_val, y_val, BATCH_SIZE),
                        epochs=EPOCHS,
                        validation_steps=X_val.shape[0] // BATCH_SIZE,
                        # TODO: add a callback to calculate predictions on the test set to plot train/val/test over epochs
                        callbacks=[early_stopping]  )
