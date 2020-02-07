import librosa
import librosa.display
import os
import numpy as np


# TODO: this script works only if you create this directory and fill it with .wav-files!
dir_test = "data"

# TODO: create output folder where the results of your preprocessing will appear as .npy file
dir_output = "Material_npy"

length_snippet_sec = 2
number_of_mel_bands = 64
# as proposed by librosa: samplerate/2
max_freq_of_mel_bands = 22050 / 2.0

def load_snippets_as_stft_matrices(path_to_files, length_snippet):
    """
    Load evenly sized snippets from .wav-files and transform each to STFT matrix

    :param path_to_files: path to .wav-files.
    :param length_snippet: length of snippets in seconds
    :return: List of snippets as short-time fourier transformation matrices
    """

    snippet_list_stft_matrices = []

    # reads every .wav-file in specified directory
    for file in os.scandir(path_to_files):
        signal, samplerate = librosa.core.load(file)
        #print(signal,samplerate)
        length_snippets_array = length_snippet * samplerate
        #print(len(signal))

        # calculate number of snippets in this file. discard last snippet if too short
        number_of_snippets = len(signal)//length_snippets_array

        # iterate over all snippets in file and calculate stft matrix for each snippet
        i = 0
        for _ in range(number_of_snippets):
            snippet = signal[length_snippets_array*i:length_snippets_array*(i+1)]
            i += 1
            stft_matrix = librosa.stft(snippet)
            snippet_list_stft_matrices.append(stft_matrix)

    return snippet_list_stft_matrices


def load_snippets_as_mel_matrices(path_to_files, length_snippet):
    """
    Load evenly sized snippets from .wav-files and transform each to mel band matrix

    :param path_to_files: path to .wav-files.
    :param length_snippet: length of snippets in seconds
    :return: List of snippets as mel band matrices
    """

    snippet_list_mel_matrices = []

    # reads every .wav-file in specified directory
    for file in os.scandir(path_to_files):
        signal, samplerate = librosa.core.load(file)
        length_snippets_array = length_snippet * samplerate
        #print("SIGNAL",signal)

        # calculate number of snippets in this file. discard last snippet if too short
        number_of_snippets = len(signal)//length_snippets_array

        # iterate over all snippets in file and calculate stft matrix for each snippet
        i = 0
        for _ in range(number_of_snippets):
            snippet = signal[length_snippets_array*i:length_snippets_array*(i+1)]
            i += 1
            mel_matrix = librosa.feature.melspectrogram(y=snippet, sr=samplerate, n_mels=number_of_mel_bands, fmax=max_freq_of_mel_bands)
            snippet_list_mel_matrices.append(mel_matrix)


    return snippet_list_mel_matrices


if __name__ == '__main__':
    stft_snippets = load_snippets_as_stft_matrices(dir_test, length_snippet_sec)
    #print(stft_snippets[0].shape)

    mel_snippets = load_snippets_as_mel_matrices(dir_test, length_snippet_sec)
    #print(mel_snippets[0].shape)

    #to print pretty
    #np.set_printoptions(suppress=True, threshold = sys.maxsize)
    #x = np.asmatrix(mel_snippets[0])
    #print(x)

    #use this function if you need list as np array
    #mel_snippets_np = np.asarray(mel_snippets)

    #write snippet list to .npy file
    np.save(dir_output + "/mel_snippets", mel_snippets)

    #load created file
    b = np.load(dir_output + "/mel_snippets.npy")
    #print(b)
