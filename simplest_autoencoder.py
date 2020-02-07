from keras.layers import Input, Dense
from keras.models import Model, load_model
import numpy as np
import matplotlib.pyplot as plt
import params
import librosa
from tensorflow.python.client import device_lib
from preprocessing import run_preprocessing
from sklearn.manifold import TSNE

from preprocessing import run_preprocessing, load_snippets_as_mel_matrices


percentage_test_data = 0.2


def train_autoencoder(x_train, x_test):
    # this is the size of our encoded representations
    encoding_dim = 30

    # this is our input placeholder
    input_img = Input(shape=(5568,))  # size of one snippet: 64 x 87 => 5568

    # "encoded" is the encoded representation of the input

    encoded = Dense(1200, activation='relu')(input_img)
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
    autoencoder = Model(input_img, decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    # create a placeholder for an encoded (32-dimensional) input
    #encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    #decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    #decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(x_train, x_train,
                    epochs=params.EPOCHS,
                    batch_size=params.BATCH_SIZE,
                    shuffle=params.SHUFFLE,
                    validation_data=(x_test, x_test))
    # save encoder, decoder
#    autoencoder.save('autoencoder_weights_decoder.h5', decoder)
#   autoencoder.save('autoencoder_weights_encoder.h5', encoder)

    autoencoder.summary()

    return encoder, autoencoder

def plot_encoded_vectors(encoded_imgs, name):
    n = 10
    plt.figure(figsize=(20, 8))
    for i in range(n):
        ax = plt.subplot(1, n, i+1)
        plt.imshow(encoded_imgs[i].reshape(10, 1 * 3).T)
        #plt.imshow(encoded_imgs[i].reshape(10, 10 * 3).T)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig("Material/pics/"+ name +"png" )
    #plt.show()

#!!! features are not interpretable, plots different every time
def tsne_presentation_of_vectors(no_leak_vectors, leak_vectors):
    #X_embedded = TSNE(n_components=2).fit_transform(vectors)
    #X_embedded.shape
    plt.figure()

    #model = TSNE(learning_rate=100)
    no_leak_transformed = TSNE(n_components=2).fit_transform(no_leak_vectors)
    x_no_leak = no_leak_transformed[:,0]
    y_no_leak = no_leak_transformed[:,1]

    leak_transformed = TSNE(n_components=2).fit_transform(leak_vectors)
    x_leak = leak_transformed[:, 0]
    y_leak = leak_transformed[:, 1]

    plt.scatter(x_no_leak,y_no_leak, c="r", label = "no leak", alpha = 0.5)
    plt.scatter(x_leak,y_leak, c="g", marker='x', label = "leak", alpha=0.5)

    plt.legend()
    plt.savefig("Material/pics/tsne.png")
    #plt.show()

def plot_encoded_decoded(x_test, decoded_imgs, plt_name):
    print("Plot of results")
    n = 5  # how many digits we will display
    plt.figure(figsize=(40, 5))
    for i in range(n):
        # display original as spectogram
        ax = plt.subplot(2, n, i + 1)
        mel_matrix = x_test[i].reshape(64, 87) # librosa.amplitude_to_db(np.abs(librosa.stft(mel_matrix)), ref=np.max) #
        librosa.display.specshow(librosa.power_to_db(mel_matrix, ref=1.0),y_axis='mel',fmax=8000,x_axis = 'time')
        print("np.amax for display specshow original")
        print(np.amax(mel_matrix))
        plt.colorbar(format='%+2.0f dB')
        ax.get_xaxis().set_visible(False) 
        ax.get_yaxis().set_visible(False)


        # display reconstruction as spectogram
        ax = plt.subplot(2, n, i + 1 + n)
        mel_matrix_dec = decoded_imgs[i].reshape(64, 87)
        librosa.display.specshow(librosa.power_to_db(mel_matrix_dec, ref=1.0),y_axis='mel',fmax=8000,x_axis = 'time')
        print("np.max for display specshow decoded")
        print(np.amax(mel_matrix_dec))
        plt.colorbar(format='%+2.0f dB')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig(plt_name+".png")
    #plt.show()

def run_simple_autoencoder():
    # load data from .npy-file and split in train and test data
    # data = np.load("Material/output/mel_snippets_test.npy")
    if params.LOAD_SNIPPED_FROM_OUTPUT:
        data = np.load(params.OUTPUT_PATH +"/"+params.SNIPPETS_NAME+".npy")
    else:
        data = run_preprocessing()

    x_train = data[:int(data.shape[0] * (1 - percentage_test_data)), :]
    print("INFO: Use data from position 0 to position ", int(data.shape[0] * (1 - percentage_test_data)),
          " as training data.")
    x_test = data[int(data.shape[0] * (1 - percentage_test_data)):, :]
    print("INFO: Use data from position ", int(data.shape[0] * (1 - percentage_test_data)), " to position ",
          data.shape[0], " as test data.")

    # reshape data from (number of samples, 64, 87) to (number of samples, 5568)
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    # train encoder and decoder oder load weights "autoencoder_weights_encoder/decoder.h5
    if params.LOAD_WEIGHTS:
        encoder = load_model('autoencoder_weights_encoder.h5')
        decoder = load_model('autoencoder_weights_decoder.h5')

    else:
        encoder, autoencoder = train_autoencoder(x_train, x_test)

    # use encoder and decoder to predict
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = autoencoder.predict(x_test)

    # plot example result
    plot_name= "Material/pics/no_leak"
    #TODO: change method name
    plot_encoded_decoded(x_test, decoded_imgs, plot_name)

    #plot encoded vectors
    name = "no_leak_vectors"
    plot_encoded_vectors(encoded_imgs, name)

    #plot tsne
    #tsne_presentation_of_vectors(decoded_imgs)

    #TODO This should probably not return x_test but for now needed in classification experiments
    return encoder, autoencoder, x_test, encoded_imgs

def test_with_leak_data(encoder, autoencoder):
    if params.LOAD_SNIPPED_FROM_OUTPUT:
        data = np.load(params.OUTPUT_PATH +"/"+params.LEAK_TEST_DATA_SNIPPETS_NAME+".npy")
    else:
        "No leakage data used for testing"
        return

    percentage_of_data_used = 0.5
    x_test = data[int(data.shape[0] * (1 - percentage_of_data_used)):, :]
    print("INFO: Use leak data")

    # reshape data from (number of samples, 64, 87) to (number of samples, 5568)
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    # use encoder and decoder to predict
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = autoencoder.predict(x_test)

    # plot example result
    plot_name = "Material/pics/leak"
    #TODO: rename method??
    plot_encoded_decoded(x_test, decoded_imgs, plot_name)

    # plot encoded vectors
    name = "leak_vectors"
    plot_encoded_vectors(encoded_imgs, name)

    print(decoded_imgs.shape)
    return encoded_imgs


if __name__ == '__main__':
    print(device_lib.list_local_devices())
    #encoder, decoder, x_test = run_simple_autoencoder()
    encoder, autoencoder, x_test, no_leak_vectors = run_simple_autoencoder()
    leak_vectors = test_with_leak_data(encoder,autoencoder)

    # plot tsne
    tsne_presentation_of_vectors(no_leak_vectors,leak_vectors)

    exit()

    # use encoder and decoder to predict
    #encoded_imgs = encoder.predict(x_test)
    #decoded_imgs = decoder.predict(encoded_imgs)

    # plot example result
    #plot_encoded_decoded(x_test, decoded_imgs)
