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
from utilities.autoencoder_utilities import load_preprocessed_snippets
import numpy as np

#train_data= load_snippets_as_mel_matrices("data",2)
#test_data=load_snippets_as_mel_matrices("data",2)
percentage_test_data = 0.2
#data = np.load("Material_npy/mel_snippets.npy")  # TODO: create this file first with preprocessing.py

#x_train = data[:int(data.shape[0]*(1-percentage_test_data)), :]

#print("Use data from position 0 to position ", int(data.shape[0] * (1 - percentage_test_data)),
  #        " as training data.")
#x_test = data[int(data.shape[0]*(1-percentage_test_data)):, :]
#print("Use data from position ", int(data.shape[0] * (1 - percentage_test_data)), " to position ",
 #         data.shape[0], " as test data.")


x_train, x_test = load_preprocessed_snippets()


original_dim=64*87
intermediate_dim=512
latent_dim=2
batch_size=32
epochs=100

x_train = x_train[:(int(x_train.shape[0]/batch_size) * batch_size)]
x_test = x_test[:(int(x_test.shape[0]/batch_size) * batch_size)]

x = Input(batch_shape=(batch_size,original_dim,))
# hidden layer
h = Dense(intermediate_dim, activation='relu')(x)
#epsilon=tf.random_normal(tf.shape(Standart_deviation_layer),dtype=tf.float32,mean=0.0,stddev=1.0)

# output layer for mean and log variance
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)
def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim))
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

def vae_loss(x, x_decoded_mean):
    xent_loss =binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return xent_loss + kl_loss

vae.compile(optimizer='rmsprop', loss=vae_loss)

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

#x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
#plt.figure(figsize=(6, 6))
#plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
#plt.colorbar()
#plt.show()


