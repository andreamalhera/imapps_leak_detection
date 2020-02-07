import params
import os
from keras.models import load_model
import numpy as np
from preprocessing import run_preprocessing
from tensorflow.python.client import device_lib

# TODO Handle case PERCENTAGE_TEST_DATA=0.2. Currently error
PERCENTAGE_TEST_DATA = 0.2

PICS_PATH = params.PICS_PATH
WEIGHTS_PATH = params.WEIGHTS_PATH
CNN_OUTPUT_PATH = params.CNN_OUTPUT_PATH
LOAD_SNIPPED_FROM_OUTPUT = params.LOAD_SNIPPED_FROM_OUTPUT
SNIPPETS_PATH = params.SNIPPETS_PATH
ENCODER_PATH = params.ENCODER_WEIGHTS_PATH
AUTOENCODER_WEIGHTS_PATH = params.AUTOENCODER_WEIGHTS_PATH


def create_directories():
    # simple autoencoder directories
    if not os.path.exists(PICS_PATH):
        os.mkdir(PICS_PATH)
    if not os.path.exists(WEIGHTS_PATH):
        os.mkdir(WEIGHTS_PATH)
    # cnn autoencoder directories
    if not os.path.exists(CNN_OUTPUT_PATH):
        os.mkdir(CNN_OUTPUT_PATH)


def init_setup():
    create_directories()

    # For GPU settup purposes
    print("INFO: Local devices list: ")
    print(device_lib.list_local_devices())


def load_preprocessed_snippets():
    # Load preprocessed input snippets
    # TODO: Unsplit train and test data and get it from preprocessing data files architecture
    if LOAD_SNIPPED_FROM_OUTPUT:
        preprocessed_snippets = np.load(SNIPPETS_PATH + ".npy")
    else:
        preprocessed_snippets = run_preprocessing()

    x_train, x_test = split_train_test_data(preprocessed_snippets)
    return x_train, x_test


def split_train_test_data(preprocessed_snippets):
    # Split snippets in training and test data
    split_position = int(preprocessed_snippets.shape[0] * (1 - PERCENTAGE_TEST_DATA))
    x_train = preprocessed_snippets[:split_position, :]
    print("INFO: Use data from position 0 to position ", split_position, " as training data.")
    x_test = preprocessed_snippets[split_position:, :]
    print("INFO: Use data from position ", split_position, " to position ",
          preprocessed_snippets.shape[0], " as test data.")
    return x_train, x_test


# TODO Fix WEIGHTS_PATH for all ae
def get_stored_model(autoencoder_name):
    # simple encoder
    # über namenübergabe die richtigen weights laden
    encoder = load_model(ENCODER_PATH[:-3] + "_" + autoencoder_name + ".h5")
    autoencoder = load_model(AUTOENCODER_WEIGHTS_PATH[:-3] + "_" + autoencoder_name + ".h5")
    print("Weights loaded for autoencoder " + autoencoder_name + " from file: " + AUTOENCODER_WEIGHTS_PATH[
                                                                                  :-3] + "_" + autoencoder_name + ".h5")
    print("Weights loaded for encoder " + autoencoder_name + " from file: " + ENCODER_PATH[
                                                                              :-3] + "_" + autoencoder_name + ".h5")

    return encoder, autoencoder


def get_autoencoder_weights_filepath(name, part, epochs, batch_size, encoding_dim, timestamp, note=None):
    epochs = str(epochs)
    batch_size = str(batch_size)
    encoding_dim = str(encoding_dim)

    if note is None:
        weights_path = "weights_" + name + "_" + part + "_ep" + epochs + "_ba" + batch_size + "_dim" + encoding_dim + "_" + timestamp + ".h5"
    else:
        weights_path = "weights_" + name + "_" + part + "_ep" + epochs + "_ba" + batch_size + "_dim" + encoding_dim + "_" + timestamp + "_" + note + ".h5"
    print("INFO: path is ", weights_path)
    return weights_path
