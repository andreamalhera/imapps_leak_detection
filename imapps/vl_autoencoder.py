# TODO: Please move to /autoencoders
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Lambda
from keras.models import Model
from keras.losses import binary_crossentropy
#from keras.utils import plot_model
from keras import backend as K
import os
import params
import  preprocessing_leak_test_data
from utilities.autoencoder_utilities import get_stored_model, load_preprocessed_snippets, create_directories

# TODO: This import should not be necessary as plotting is part of testing
from utilities import test_utilities

import numpy as np



#train_data= load_snippets_as_mel_matrices("data",2)
#test_data=load_snippets_as_mel_matrices("data",2)
percentage_test_data = 0.2
#data = np.load("data/output/mel_snippets.npy")  # TODO: create this file first with preprocessing.py


create_directories()
x_train, x_test = load_preprocessed_snippets()


#print(data)
#x_train = data[:int(data.shape[0]*(1-percentage_test_data)), :]


#print("Use data from position 0 to position ", int(data.shape[0] * (1 - percentage_test_data))," as training data.")
#x_test = data[int(data.shape[0]*(1-percentage_test_data)):, :]
#print("Use data from position ", int(data.shape[0] * (1 - percentage_test_data)), " to position ",data.shape[0], " as test data.")

x_train, x_test = load_preprocessed_snippets()


AUTOENCODER_NAME="vae"
LOAD_WEIGHTS=False
original_dim=64*87
intermediate_dim=512
latent_dim=2
batch_size=32
epochs=1
input_shape = (original_dim, )
epsilon_std=1.


x_train = x_train[:(int(x_train.shape[0]/batch_size) * batch_size)]
x_test = x_test[:(int(x_test.shape[0]/batch_size) * batch_size)]

#x = Input(shape=input_shape,name="encoder_input")

x=Input(batch_shape=(batch_size,original_dim))
h = Dense(intermediate_dim, activation='relu')(x)



z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)



def sampling(args):
    z_mean, z_log_sigma = args
    batch = K.shape(z_mean)[0]

    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch_size,latent_dim))
    return z_mean + K.exp(z_log_sigma) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])



decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

#end-to-end autoencoder
vae = Model(x, x_decoded_mean)


def vae_loss (y_true,y_pred):


    xent_loss =K.sum(K.binary_crossentropy(y_true, y_pred))
    kl_loss = - 0.5 * K.mean(1+ z_log_sigma- K.square(z_mean)-K.exp(z_log_sigma),axis=-1)
    return xent_loss + kl_loss

vae.compile(optimizer='rmsprop', loss=vae_loss)


#max_value=float(x_train.max())

#x_train = np.reshape(x_train, [-1, original_dim])
#x_test = np.reshape(x_test, [-1, original_dim])



#x_train, x_test = load_preprocessed_snippets()
max_value=float(x_train.max())
x_train = x_train.astype('float32') / max_value
x_test = x_test.astype('float32') / max_value

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
vae.fit(x_train, x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test))


# encoder, from inputs to latent space
encoder = Model(x, z_mean)

# generator, from latent space to reconstructed inputs
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

vae.save(params.AUTOENCODER_WEIGHTS_PATH, vae.sample_weights)
vae.save(params.ENCODER_WEIGHTS_PATH, encoder)
encoded_imgs = encoder.predict(x_test,batch_size=batch_size)
decoded_imgs = generator.predict(encoded_imgs)
#print(encoded_imgs.shape)
#print(encoded_imgs.shape[1])
#print(encoded_imgs.shape[2])

def predict_varational_autoencoder(encoder, decoder, x_test,latent_dim=2):
    # use encoder and decoder to predict

    encoded_imgs = encoder.predict(x_test,batch_size=batch_size)
    #encoded_imgs=encoded_imgs[1]
    decoded_imgs = decoder.predict(encoded_imgs)
    return encoded_imgs, decoded_imgs

def run_simple_plots(x_true, x_encoded, x_pred):
    # plot example result
    test_utilities.plot_encoded_decoded_simplest(x_true, x_pred, params.PLOT_NO_LEAK_PATH)
    #plot encoded vectors
    test_utilities.plot_encoded_vectors(x_encoded, params.PLOT_NO_LEAK_NAME)
    #plot tsne
    #tsne_presentation_of_vectors(x_pred)
#print(run_simple_plots(x_test,encoded_imgs,decoded_imgs))
if LOAD_WEIGHTS:
    encoder, decoder = get_stored_model(AUTOENCODER_NAME)
else:
    encoder, decoder =encoder,generator
    encoded_imgs, decoded_img=predict_varational_autoencoder(encoder, decoder, x_test)






