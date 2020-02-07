import matplotlib.pyplot as plt
import librosa
import numpy as np
import params
from sklearn.manifold import TSNE

NUM_OF_VECTORS = 5
NUM_OF_DISPLAY_DIGITS=5

PICS_PATH=params.PICS_PATH
TNSE_PATH=params.TNSE_PATH

# use if embedding dimensionality is only 2
def plot_2D_vectors(no_leak_vectors, leak_vectors, path):
    plt.figure()

    x_no_leak = no_leak_vectors[:,0]
    y_no_leak = no_leak_vectors[:,1]

    #leak_transformed = TSNE(n_components=2).fit_transform(leak_vectors)
    x_leak = leak_vectors[:, 0]
    y_leak = leak_vectors[:, 1]

    plt.scatter(x_no_leak,y_no_leak, c="r", label = "no leak", alpha = 0.5)
    plt.scatter(x_leak,y_leak, c="g", marker='x', label = "leak", alpha=0.5)

    plt.legend()
    plt.savefig(path + "2D_encoded_vectors.png")


# use this function for cnn autoenc
def plot_encoded_decoded(x_test, decoded_imgs, name):
    plt.figure(figsize=(40, 5))
    for i in range(NUM_OF_DISPLAY_DIGITS):
        # display original as spectogram
        ax = plt.subplot(2, NUM_OF_DISPLAY_DIGITS, i + 1)
        mel_matrix = x_test[i].reshape(64, 87) # librosa.amplitude_to_db(np.abs(librosa.stft(mel_matrix)), ref=np.max) #
        librosa.display.specshow(librosa.power_to_db(mel_matrix, ref=1.0),y_axis='mel',fmax=8000,x_axis = 'time')
        print("np.amax for display specshow original")
        print(np.amax(mel_matrix))
        plt.colorbar(format='%+2.0f dB')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


        # display reconstruction as spectogram
        ax = plt.subplot(2, NUM_OF_DISPLAY_DIGITS, i + 1 + NUM_OF_DISPLAY_DIGITS)
        mel_matrix_dec = decoded_imgs[i].reshape(64, 87)
        librosa.display.specshow(librosa.power_to_db(mel_matrix_dec, ref=1.0),y_axis='mel',fmax=8000,x_axis = 'time')
        print("np.max for display specshow decoded")
        print(np.amax(mel_matrix_dec))
        plt.colorbar(format='%+2.0f dB')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    #plot_path = os.path.join(path, 'plot.png')
    plt.savefig(name)

# use this function for simplest autoencoder
def plot_encoded_decoded_simplest(x_test, decoded_imgs, plt_name):
    print("Plot of results")
    plt.figure(figsize=(40, 5))
    for i in range(NUM_OF_DISPLAY_DIGITS):
        # display original as spectogram
        ax = plt.subplot(2, NUM_OF_DISPLAY_DIGITS, i + 1)
        mel_matrix = x_test[i].reshape(64, 87) # librosa.amplitude_to_db(np.abs(librosa.stft(mel_matrix)), ref=np.max) #
        librosa.display.specshow(librosa.power_to_db(mel_matrix, ref=1.0),y_axis='mel',fmax=8000,x_axis = 'time')
        print("np.amax for display specshow original")
        print(np.amax(mel_matrix))
        plt.colorbar(format='%+2.0f dB')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


        # display reconstruction as spectogram
        ax = plt.subplot(2, NUM_OF_DISPLAY_DIGITS, i + 1 + NUM_OF_DISPLAY_DIGITS)
        mel_matrix_dec = decoded_imgs[i].reshape(64, 87)
        librosa.display.specshow(librosa.power_to_db(mel_matrix_dec, ref=1.0),y_axis='mel',fmax=8000,x_axis = 'time')
        print("np.max for display specshow decoded")
        print(np.amax(mel_matrix_dec))
        plt.colorbar(format='%+2.0f dB')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig(plt_name)
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
    plt.savefig(TNSE_PATH)
    #plt.show()


# use only for simplest autoencoder with encoding dimensionality of 30 or 300
def plot_encoded_vectors(encoded_imgs, name):
    plt.figure(figsize=(20, 8))
    for i in range(NUM_OF_VECTORS):
        ax = plt.subplot(1, NUM_OF_VECTORS, i+1)
        plt.imshow(encoded_imgs[i].reshape(10, 1 * 3).T)
        #plt.imshow(encoded_imgs[i].reshape(10, 10 * 3).T)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(PICS_PATH+name+"_vectors")
    #plt.show()
