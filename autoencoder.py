# TODO: Please move to /autoencoders
from keras.models import Model, Sequential
from keras.layers import Dense, Input
from audio_playground import load_snippets_as_mel_matrices
import numpy as np
import matplotlib.pyplot as plt

train_data= load_snippets_as_mel_matrices("data",2)
test_data=load_snippets_as_mel_matrices("data",2)
x_train =np.array(train_data)
x_test = np.array(test_data)

#print(x_train.shape[1])

# Scales the training and test data to range between 0 and 1.
max_value=float(x_train.max())
x_train = x_train.astype('float32') / max_value
x_test = x_test.astype('float32') / max_value


# input dimension = 5568

encoding_dim = 32
input_dim= np.prod(x_train.shape[1:])


x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

autoencoder=Sequential()
#autoencoder.add(Dense(encoding_dim, input_shape=(input_dim,),activation="relu"))
#autoencoder.add(Dense(input_dim,activation="sigmoid"))
#x_dim=Input(shape=(input_dim,))
#encoder_layer=autoencoder.layers[0]
#encoder=Model(x_dim,encoder_layer(x_dim))

#Encoder Layers
autoencoder.add(Dense(4 * encoding_dim, input_shape=(input_dim,), activation='relu'))
autoencoder.add(Dense(2 * encoding_dim, activation='relu'))
autoencoder.add(Dense(encoding_dim, activation='relu'))

# Decoder Layers
autoencoder.add(Dense(2 * encoding_dim, activation='relu'))
autoencoder.add(Dense(4 * encoding_dim, activation='relu'))
autoencoder.add(Dense(input_dim, activation='sigmoid'))

x_input=Input(shape=(input_dim,))
encoder_layer1 = autoencoder.layers[0]
encoder_layer2 = autoencoder.layers[1]
encoder_layer3 = autoencoder.layers[2]
encoder = Model(x_input, encoder_layer3(encoder_layer2(encoder_layer1(x_input))))

#encoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                validation_data=(x_test, x_test))

num_data = 5
np.random.seed(42)
random_test_data = np.random.randint(x_test.shape[0], size=num_data)
#print(random_test_images)

encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

plt.figure(figsize=(18, 4))


for i, data_idx in enumerate(random_test_data):
    # plot original image
    ax = plt.subplot(3, num_data, i + 1)
    plt.plot(x_test[data_idx].reshape(64, 87))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    # plot encoded image
    ax = plt.subplot(3, num_data, num_data + i + 1)
    plt.plot(encoded_imgs[data_idx].reshape(8, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    # plot reconstructed image
    ax = plt.subplot(3, num_data, 2 * num_data + i + 1)
    plt.plot(decoded_imgs[data_idx].reshape(64, 87))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()




