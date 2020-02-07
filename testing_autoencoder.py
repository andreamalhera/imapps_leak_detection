import numpy as np
import params
import preprocessing_leak_test_data
from cnn_autoencoder import run_cnn_autoencoder, predict_cnn_autoencoder
from utilities import test_utilities
from simple_autoencoder import run_simple_autoencoder, predict_simple_autoencoder
from utilities.autoencoder_utilities import load_preprocessed_snippets

from utilities.test_utilities import plot_2D_vectors, plot_encoded_decoded, \
    plot_encoded_decoded_simplest, plot_encoded_vectors, tsne_presentation_of_vectors

# TODO: Import all tests from all autoencoders

PLOT_ACTIVATION = False
PERCENTAGE_DATA_USED = 0.5

LOAD_LEAK_SNIPPEDS_FROM_OUTPUT=params.LOAD_LEAK_SNIPPEDS_FROM_OUTPUT
LEAK_TEST_DATA_SNIPPETS_PATH=params.LEAK_TEST_DATA_SNIPPETS_PATH
PLOT_LEAK_PATH=params.PLOT_LEAK_PATH
PLOT_LEAK_NAME=params.PLOT_LEAK_NAME
PLOT_NO_LEAK_PATH=params.PLOT_NO_LEAK_PATH
PLOT_NO_LEAK_NAME=params.PLOT_NO_LEAK_NAME
CNN_OUTPUT_PATH=params.CNN_OUTPUT_PATH

def test_with_leak_data(encoder, autoencoder):
    if LOAD_LEAK_SNIPPEDS_FROM_OUTPUT:
        data = np.load(LEAK_TEST_DATA_SNIPPETS_PATH + ".npy")
    else:
        # "No leakage data used for testing"
        data = preprocessing_leak_test_data.run_preprocessing()

    x_test = data[int(data.shape[0] * (1 - PERCENTAGE_DATA_USED)):, :]

    print("INFO: Use leak data")

    # reshape data from (number of samples, 64, 87) to (number of samples, 5568)
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    # use encoder and decoder to predict
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = autoencoder.predict(x_test)

    if PLOT_ACTIVATION:
        # plot example result
        test_utilities.plot_encoded_decoded_simplest(x_test, decoded_imgs, PLOT_LEAK_PATH)

        # plot encoded vectors
        test_utilities.plot_encoded_vectors(encoded_imgs, PLOT_LEAK_NAME)

    print(decoded_imgs.shape)
    return encoded_imgs


def test_with_leak_data_cnn(x_test, encoder, autoencoder):
    print("INFO: Use leak data")

    # use encoder and decoder to predict
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = autoencoder.predict(x_test)

    # plot example result
    plot_name = CNN_OUTPUT_PATH + "/leak"
    test_utilities.plot_encoded_decoded(x_test, decoded_imgs, plot_name)

    # plot encoded vectors
    name = "leak_vectors"
    # plot_encoded_vectors(encoded_imgs, name)

    print(decoded_imgs.shape)
    return encoded_imgs


def run_simple_plots(x_true, x_encoded, x_pred):
    # plot example result
    test_utilities.plot_encoded_decoded_simplest(x_true, x_pred, PLOT_NO_LEAK_PATH)
    # plot encoded vectors
    test_utilities.plot_encoded_vectors(x_encoded, PLOT_NO_LEAK_NAME)
    # plot tsne
    # tsne_presentation_of_vectors(x_pred)


def run_test_simple(x_test):
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    encoder, autoencoder = run_simple_autoencoder()
    test_with_leak_data(encoder, autoencoder)
    encoded_imgs, decoded_imgs = predict_simple_autoencoder(encoder, autoencoder, x_test)
    if PLOT_ACTIVATION:
        run_simple_plots(x_test, encoded_imgs, decoded_imgs)

def run_cnn_plots(x_true, x_encoded, x_pred):
    test_utilities.plot_encoded_decoded(x_true, x_pred,
                                       CNN_OUTPUT_PATH +"/no_leak")


def run_test_cnn(x_test):
    x_test = np.reshape(x_test, (-1, x_train.shape[1], x_train.shape[2], 1))
    encoder, decoder, autoencoder = run_cnn_autoencoder()
    leak_vectors = test_with_leak_data_cnn(x_test, encoder, autoencoder)
    encoded_imgs, decoded_imgs = predict_cnn_autoencoder(encoder, decoder, x_test)
    test_utilities.plot_2D_vectors(encoded_imgs, leak_vectors, CNN_OUTPUT_PATH + "/leak")
    if PLOT_ACTIVATION:
        # plot example result
        run_cnn_plots(x_test, encoded_imgs, decoded_imgs)


if __name__ == '__main__':
    x_train, x_test = load_preprocessed_snippets()
    run_test_simple(x_test)
    run_test_cnn(x_test)
