from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D
from keras.layers import Flatten, Reshape
from keras.models import Model
import numpy as np
import argparse
import datetime
import os
import params
from utilities.autoencoder_utilities import get_stored_model, load_preprocessed_snippets, init_setup

# TODO this import should not be necessary here because plotting should only be part of testing
from utilities import test_utilities

ENCODING_DIM_CNN = 30
LOAD_WEIGHTS = False
AUTOENCODER_NAME = "cnn"
SHUFFLE = True
EPOCHS = 30
BATCH_SIZE = 32
LOSS_CNN = "binary_crossentropy"  # binary_crossentropy, mean_squared_error
CNN_SHAPE = [8, 11, 8]

WEIGHTS_PATH = params.WEIGHTS_PATH


def encoding_layers(input_matrix, encoding_dim, num_convs):
    x = ZeroPadding2D(padding=((0, 0), (0, 1)))(input_matrix)
    for i in range(num_convs):
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    for i in range(num_convs):
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    for i in range(num_convs):
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    encoded = Dense(encoding_dim)(x)
    return encoded


def decoding_layers(encoded, encoding_dim, num_convs):
    x = Dense((CNN_SHAPE[0] * CNN_SHAPE[1] * CNN_SHAPE[2]))(encoded)
    x = Reshape(CNN_SHAPE)(x)
    for i in range(num_convs):
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    for i in range(num_convs):
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    for i in range(num_convs):
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    # Crop (top, bottom), (left, right)
    decoded = Cropping2D(cropping=((0, 0), (0, 1)))(x)
    return decoded


def train_cnn_autoencoder(x_train, x_test, encoding_dim=ENCODING_DIM_CNN, num_convs=1):
    # reshape data from (number of samples, 64, 87) to (number of samples, 64, 87, 1)
    x_train = np.reshape(x_train, (-1, x_train.shape[1], x_train.shape[2], 1))
    x_test = np.reshape(x_test, (-1, x_train.shape[1], x_train.shape[2], 1))

    # this is our input placeholder. Shape needed for first layer.
    input_shape = (x_train.shape[1], x_train.shape[2], 1)
    input_training = Input(shape=input_shape)

    # Encoded layers
    encoded = encoding_layers(input_training, encoding_dim, num_convs)
    # Decoded layers
    decoded = decoding_layers(encoded, encoding_dim, num_convs)

    autoencoder = Model(input_training, decoded)
    autoencoder.compile(optimizer='adadelta', loss=LOSS_CNN)
    autoencoder.fit(x_train, x_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    shuffle=SHUFFLE,
                    validation_data=(x_test, x_test))

    # Output model structure
    autoencoder.summary()
    now = datetime.datetime.now()
    weight_path_autoencoder = os.path.join(WEIGHTS_PATH, "CNN_weights_e30_dim30_ba32" + now.strftime("_%Y-%m-%d-%H:%M:%S") + ".h5")
    autoencoder.save(weight_path_autoencoder)

    # create the encoder model
    encoder = Model(input_training, encoded)
    weight_path_encoder = os.path.join(WEIGHTS_PATH, "CNN_weights_encoder_e30_dim30_ba32" + now.strftime("_%Y-%m-%d-%H:%M:%S") + ".h5")
    # TODO: weights are not saved correctly
    # encoder.save_weights(weight_path_encoder)
    autoencoder.save(weight_path_encoder, encoder)

    # create the decoder model
    encoded_input = Input(shape=encoding_layers(input_training, encoding_dim, num_convs).shape[1:].as_list())
    decoder = Model(encoded_input, decoding_layers(encoded_input, encoding_dim, num_convs))

    return encoder, decoder, autoencoder


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Load trained model and use it for predictions'
    )
    parser.add_argument('-d', '--cnn_encoding_dim', type=int, default=2)
    parser.add_argument('-c', '--num_convs', type=int, default=1)

    return parser.parse_args(args)


def predict_cnn_autoencoder(encoder, decoder, x_test):
    # use encoder and decoder to predict
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    return encoded_imgs, decoded_imgs


def run_cnn_autoencoder():
    args = parse_args()
    encoding_dim = args.cnn_encoding_dim
    num_convs = args.num_convs

    x_train, x_test = load_preprocessed_snippets()
    # TODO: testing should not happen here
    x_test = np.reshape(x_test, (-1, x_train.shape[1], x_train.shape[2], 1))

    # TODO: Fix load_weights=True. Weights cannot be saved correctly and therefore also not read
    if LOAD_WEIGHTS:
        encoder, autoencoder = get_stored_model(AUTOENCODER_NAME)
    else:
        # train encoder and decoder
        encoder, decoder, autoencoder = train_cnn_autoencoder(x_train, x_test, encoding_dim, num_convs)

    # TODO: the rest of this class is testing
    encoded_imgs, decoded_imgs = predict_cnn_autoencoder(encoder, decoder, x_test)
    return encoder, decoder, autoencoder


if __name__ == '__main__':
    init_setup()
    run_cnn_autoencoder()
    exit()
