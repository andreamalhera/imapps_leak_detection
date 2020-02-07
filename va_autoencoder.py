# TODO: Please move to /autoencoders
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import params
import tensorflow as tf
from keras.layers import Dense, Input
from keras.layers import Lambda
from keras.models import Model
from keras.losses import binary_crossentropy
#from keras.utils import plot_model
from keras import backend as K
from utilities import test_utilities
from keras.models import load_model


from utilities.autoencoder_utilities import load_preprocessed_snippets, init_setup, get_stored_model
import numpy as np


print("Avalible:",tf.test.is_gpu_available())
ENCODING_DIM=30
LOAD_WEIGHTS=False
AUTOENCODER_NAME= "simple"
SHUFFLE = True
EPOCHS = 10
BATCH_SIZE = 32
LOSS_SIMPLE_AUTOENCODER = "mean_squared_error"   #binary_crossentropy, mean_squared_error
PLOT_ACTIVATION=False
AUTOENCODER_WEIGHTS_PATH=params.AUTOENCODER_WEIGHTS_PATH
ENCODER_PATH=params.ENCODER_WEIGHTS_PATH_VAE

percentage_test_data = 0.2
#data = np.load("data/output/cnn/mel_snippets.npy")  # TODO: create this file first with preprocessing.py


original_dim= 64*87
intermediate_dim= 512
latent_dim= 2
batch_size= 32
epochs= 1


def predict_va_autoencoder(vae, x_test):
    # use encoder and decoder to predict
    decoded_imgs = vae.predict(x_test)
    return decoded_imgs


def train_va_autoencoder(x_train, x_test, encoding_dim=ENCODING_DIM):
    x_train = x_train[:(int(x_train.shape[0] / BATCH_SIZE) * BATCH_SIZE)]
    x_test = x_test[:(int(x_test.shape[0] / BATCH_SIZE) * BATCH_SIZE)]

    x = Input(batch_shape=(BATCH_SIZE, original_dim,))
    # hidden layer
    h = Dense(intermediate_dim, activation='relu')(x)
    # epsilon=tf.random_normal(tf.shape(Standart_deviation_layer),dtype=tf.float32,mean=0.0,stddev=1.0)

    # output layer for mean and log variance
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(BATCH_SIZE, latent_dim))
        return z_mean + K.exp(z_log_sigma) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # end-to-end autoencoder
    vae = Model(x, x_decoded_mean)

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    def vae_loss(x, x_decoded_mean):
        xent_loss = binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        return xent_loss + kl_loss

    vae.compile(optimizer='rmsprop', loss=vae_loss)

    max_value = float(x_train.max())
    x_train = x_train.astype('float32') / max_value
    x_test = x_test.astype('float32') / max_value

    x_test = np.reshape(x_test, (-1, x_train.shape[1], x_train.shape[2], 1))

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    vae.fit(x_train, x_train,
            shuffle=True,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, x_test))

    # x_test, decoded_imgs, params.PLOT_LEAK_PATH
    #test_utilities.plot_encoded_decoded(x_test, vae, params.CNN_OUTPUT_PATH + "/no_leak_VAE")

    vae.summary()
    #TODO: change path
    vae.save(ENCODER_PATH+"vae_autoencoder_weights.h5", vae.sample_weights)

    return vae


def run_va_autoencoder():
    x_train, x_test = load_preprocessed_snippets()
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    # x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    # train encoder and decoder oder load weights "autoencoder_weights_encoder/decoder.h5
    if LOAD_WEIGHTS:
      #  vae = get_stored_model(AUTOENCODER_NAME)
        vae = load_model(ENCODER_PATH+"vae_autoencoder_weights.h5")

    else:
        vae = train_va_autoencoder(x_train, x_test)
  #TODO
    print(x_test.shape)
  #  encoded_imgs, decoded_imgs = predict_va_autoencoder(vae, x_test)

    return vae

if __name__ == '__main__':
    init_setup()
    run_va_autoencoder()