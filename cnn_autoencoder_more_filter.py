from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D
from keras.layers import Flatten, Reshape
from keras.models import Model
import numpy as np
import argparse
import os
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import datetime
import params
import preprocessing_leak_test_data
from utilities.autoencoder_utilities import get_stored_model, load_preprocessed_snippets, init_setup

# TODO this import should not be necessary here because plotting should only be part of testing
from utilities import test_utilities

ENCODING_DIM_CNN = 128
LOAD_WEIGHTS = False
AUTOENCODER_NAME = "cnn_more_filter"
SHUFFLE = True
EPOCHS = 50
BATCH_SIZE = 64
LOSS_CNN = "mean_squared_error"  # binary_crossentropy, mean_squared_error
CNN_SHAPE = [2, 3, 128]

WEIGHTS_PATH = params.WEIGHTS_PATH


def encoding_layers(input_matrix, encoding_dim):
    x = ZeroPadding2D(padding=((0, 0), (0, 1)))(input_matrix)
    #x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    #x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    #x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    #x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = ZeroPadding2D(padding=((0, 0), (0, 1)))(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    #x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    encoded = Dense(encoding_dim)(x)
    return encoded


def decoding_layers(encoded, encoding_dim):
    x = Dense((CNN_SHAPE[0] * CNN_SHAPE[1] * CNN_SHAPE[2]))(encoded)
    x = Reshape(CNN_SHAPE)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    #x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Cropping2D(cropping=((0, 0), (0, 1)))(x)
    #x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    #x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    #x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    #x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(1, (3, 3), activation=None, padding='same')(x)
    # Crop (top, bottom), (left, right)
    decoded = Cropping2D(cropping=((0, 0), (0, 1)))(x)
    return decoded


def train_cnn_autoencoder(x_train, x_test, encoding_dim=ENCODING_DIM_CNN):
    # reshape data from (number of samples, 64, 87) to (number of samples, 64, 87, 1)
    x_train = np.reshape(x_train, (-1, x_train.shape[1], x_train.shape[2], 1))
    x_test = np.reshape(x_test, (-1, x_train.shape[1], x_train.shape[2], 1))

    # this is our input placeholder. Shape needed for first layer.
    input_shape = (x_train.shape[1], x_train.shape[2], 1)
    input_training = Input(shape=input_shape)

    # Encoded layers
    encoded = encoding_layers(input_training, encoding_dim)
    # Decoded layers
    decoded = decoding_layers(encoded, encoding_dim)

    autoencoder = Model(input_training, decoded)
    autoencoder.compile(optimizer='adam', loss=LOSS_CNN)
    autoencoder.fit(x_train, x_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    shuffle=SHUFFLE,
                    validation_data=(x_test, x_test))

    # Output model structure
    autoencoder.summary()
    now = datetime.datetime.now()
    weight_path_autoencoder = os.path.join(WEIGHTS_PATH, "CNN_DEEP_weights_e50_dim128_ba64" + now.strftime("_%Y-%m-%d-%H:%M:%S") + ".h5")
    autoencoder.save(weight_path_autoencoder)

    # create the encoder model
    encoder = Model(input_training, encoded)
    weight_path_encoder = os.path.join(WEIGHTS_PATH, "CNN_DEEP_weights_encoder_e50_dim128_ba64" + now.strftime("_%Y-%m-%d-%H:%M:%S") + ".h5")
    # Now it saves whole model(structure + weights)
    autoencoder.save(weight_path_encoder, encoder)
    print("cnn saved")

    # create the decoder model
    encoded_input = Input(shape=encoding_layers(input_training, encoding_dim).shape[1:].as_list())
    decoder = Model(encoded_input, decoding_layers(encoded_input, encoding_dim))

    return encoder, decoder, autoencoder


# TODO: this function should be part of testing
def test_with_leak_data(encoder, autoencoder):
    if params.LOAD_SNIPPED_FROM_OUTPUT:
        data = np.load(params.LEAK_TEST_DATA_SNIPPETS_PATH + ".npy")
    else:
        "No leakage data used for testing"
        data = preprocessing_leak_test_data.run_preprocessing()

    # percentage_of_data_used = 0.5
    x_test = data  # [int(data.shape[0] * (1 - percentage_of_data_used)):, :]
    print("INFO: Use leak data")

    input_shape = (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    # reshape data
    x_test = x_test.reshape(input_shape)

    # use encoder and decoder to predict
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = autoencoder.predict(x_test)

    # plot example result
    plot_name = params.CNN_MORE_FILTER_OUTPUT_PATH + "/leak"
    test_utilities.plot_encoded_decoded(x_test, decoded_imgs, plot_name)

    # plot encoded vectors
    name = "leak_vectors"
    # plot_encoded_vectors(encoded_imgs, name)

    print(decoded_imgs.shape)
    return encoded_imgs


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Load trained model and use it for predictions'
    )
    parser.add_argument('-d', '--cnn_encoding_dim', type=int, default=2)

    return parser.parse_args(args)


def predict_cnn_autoencoder(encoder, decoder, x_test):
    # use encoder and decoder to predict
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    return encoded_imgs, decoded_imgs


def run_cnn_plots(x_true, x_encoded, x_pred):
    test_utilities.plot_encoded_decoded(x_true, x_pred,
                                        params.CNN_MORE_FILTER_OUTPUT_PATH + "/no_leak")


def run_cnn_autoencoder():
    args = parse_args()
    encoding_dim = args.cnn_encoding_dim

    x_train, x_test = load_preprocessed_snippets()
    # TODO: testing should not happen here
    x_test = np.reshape(x_test, (-1, x_train.shape[1], x_train.shape[2], 1))

    # TODO: Fix load_weights=True. Weights cannot be saved correctly and therefore also not read
    if LOAD_WEIGHTS:
        encoder, autoencoder = get_stored_model(AUTOENCODER_NAME)
    else:
        # train encoder and decoder
        encoder, decoder, autoencoder = train_cnn_autoencoder(x_train, x_test, encoding_dim)

    # TODO: the rest of this class is testing
    #encoded_imgs, decoded_imgs = predict_cnn_autoencoder(encoder, decoder, x_test)

    # plot example result
    # run_cnn_plots(x_test, encoded_imgs, decoded_imgs)

    #leak_vectors = test_with_leak_data(encoder, autoencoder)
    #test_utilities.plot_2D_vectors(encoded_imgs, leak_vectors, params.CNN_MORE_FILTER_OUTPUT_PATH + "/leak")


if __name__ == '__main__':
    init_setup()

    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    #config.log_device_placement = True  # to log device placement (on which device the operation ran)
    #sess = tf.Session(config=config)
    #set_session(sess)  # set this TensorFlow session as the default session for Keras

    run_cnn_autoencoder()
    exit()
