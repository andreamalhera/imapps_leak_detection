from keras.layers import Input, Dense
from keras.models import Model
from keras import metrics
import numpy as np
import params
import tensorflow as tf
#TODO: import PCA and try out
from utilities.autoencoder_utilities import get_stored_model, load_preprocessed_snippets, init_setup

# TODO: This import should not be necessary as plotting is part of testing
from utilities import test_utilities

print("Avalible:",tf.test.is_gpu_available())
ENCODING_DIM=30
LOAD_WEIGHTS=False
AUTOENCODER_NAME= "simple"
SHUFFLE = True
EPOCHS = 10
BATCH_SIZE = 64
LOSS_SIMPLE_AUTOENCODER = "mean_squared_error"   #binary_crossentropy, mean_squared_error
PLOT_ACTIVATION=False

def train_simple_autoencoder(x_train, x_test, encoding_dim=ENCODING_DIM):
    # reshape data from (number of samples, 64, 87) to (number of samples, 5568)
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    # this is our input placeholder
    input_training = Input(shape=(5568,))  # size of one snippet: 64 x 87 => 5568

    # "encoded" is the encoded representation of the input
    encoded = Dense(1200, activation='relu')(input_training)
    encoded = Dense(600, activation='relu')(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)

    # "decoded" is the lossy reconstruction of the input
    #decoded = Dense(5568, activation='sigmoid')(encoded)
    decoded = Dense(600, activation='relu')(encoded)
    decoded = Dense(1200, activation='relu')(decoded)
    #decoded = Dense(300, activation='relu')(decoded)
    decoded = Dense(5568, activation='sigmoid')(decoded)
    #decoded = Dense(units=784, activation='sigmoid')(decoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_training, decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_training, encoded)

    # create a placeholder for an encoded (32-dimensional) input
    #encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    #decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    #decoder = Model(encoded_input, decoder_layer(encoded_input))
    #mean_squared_error, binary_crossentropy
    autoencoder.compile(optimizer='adadelta', loss=LOSS_SIMPLE_AUTOENCODER,
                        metrics=[metrics.mae])


    autoencoder.fit(x_train, x_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    shuffle=SHUFFLE,
                    validation_data=(x_test, x_test))


    # collect losses
    x_concat = np.concatenate([x_test, x_train], axis=0)
    losses = []

    for x in x_concat:
        # compule loss for each test sample
        x = np.expand_dims(x, axis=0)
        loss = autoencoder.test_on_batch(x, x)
        losses.append(loss)
    #plot_losses(losses)

    # save encoder, decoder

    autoencoder.summary()
    autoencoder.save(params.AUTOENCODER_WEIGHTS_PATH, autoencoder.sample_weights)
    autoencoder.save(params.ENCODER_WEIGHTS_PATH, encoder)

    return encoder, autoencoder

def predict_simple_autoencoder(encoder, autoencoder, x_test):
    # use encoder and decoder to predict
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = autoencoder.predict(x_test)
    return encoded_imgs, decoded_imgs

def run_simple_autoencoder():
    x_train, x_test = load_preprocessed_snippets()
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    # train encoder and decoder oder load weights "autoencoder_weights_encoder/decoder.h5
    if LOAD_WEIGHTS:
        encoder, autoencoder = get_stored_model(AUTOENCODER_NAME)
    else:
        encoder, autoencoder = train_simple_autoencoder(x_train, x_test)
    encoded_imgs, decoded_imgs =predict_simple_autoencoder(encoder, autoencoder, x_test)

    return encoder, autoencoder


if __name__ == '__main__':
    init_setup()
    run_simple_autoencoder()
    exit()
