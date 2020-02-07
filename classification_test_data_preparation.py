import params
import numpy as np
import os
import model_data_preparation
import librosa

"""
This script preprocesses data so it can be used to evaluate how good a trained model can classify into anomaly and
non-anomaly. For this purpose all .wav-files which are NOT already used to train the autoencoders get preprocessed
and saved in .npy-files. The file names are saved in lists, one for data with leaks and one for data without.
These lists can later be used to confirm classifications and as "blacklists" to exclude files never to be used for training.
"""

CHALLENGE = False #set true if unlabelled data from challenge_data folder should be used
CLASSIFICATION_DATA_PATH = "data/classification_test_data"
MODEL_DATA_FOLDER = "data/model_data"
TEST_PERCENTAGE = 0.2
SAMPLERATE= params.SAMPLERATE
NUMBER_OF_MEL_BANDS=params.NUMBER_OF_MEL_BANDS
MAX_FREQ_MEL_BAND=params.MAX_FREQ_MEL_BAND
SNIPPET_LENGTH=params.SNIPPET_LENGTH

# blacklist for files never to be used for training
no_leak_list = []
leak_list = []
unknown = []


def create_directory():
    if not os.path.exists(CLASSIFICATION_DATA_PATH):
        os.mkdir(CLASSIFICATION_DATA_PATH)
    if not os.path.exists(CLASSIFICATION_DATA_PATH + "/output-challenge-npy-files"):
        os.mkdir(CLASSIFICATION_DATA_PATH + "/output-challenge-npy-files")
    if not os.path.exists(CLASSIFICATION_DATA_PATH + "/output-labelled-npy-files"):
        os.mkdir(CLASSIFICATION_DATA_PATH + "/output-labelled-npy-files")
    if not os.path.exists(CLASSIFICATION_DATA_PATH + "/challenge_data"):
        os.mkdir(CLASSIFICATION_DATA_PATH + "/challenge_data")
    if not os.path.exists(CLASSIFICATION_DATA_PATH + "/leak_eval_data"):
        os.mkdir(CLASSIFICATION_DATA_PATH + "/leak_eval_data")
    if not os.path.exists(CLASSIFICATION_DATA_PATH + "/no_leak_eval_data"):
        os.mkdir(CLASSIFICATION_DATA_PATH + "/no_leak_eval_data")


def load_test_data():
    # load data from .npy-file
    data = np.load(MODEL_DATA_FOLDER + "test_data.npy")
    return data


def save_as_npy(dataset, path):
    # write mel snippet list to .npy file
    np.save(path + ".npy", dataset)
    print("INFO: Snippets saved in " + path + ".npy")
    return


# looks like the method in preprocessing but does some things different
def load_snippets_as_mel_matrices(path_to_files, length_snippet, blacklist):
    """
    Load evenly sized snippets from .wav-files and transform each to mel band matrix

    :param path_to_files: path to .wav-files.
    :param length_snippet: length of snippets in seconds
    :return: List of snippets as mel band matrices
    """

    #snippet_list_mel_matrices = []
    # reads every .wav-file in specified directory
    count_data = 0
    size_directory = 0
    for file in os.scandir(path_to_files):
        size_directory = size_directory+1
    for file in os.scandir(path_to_files):
        blacklist.append(file.name[:-4])
        snippet_list_mel_matrices = []
        if 'Store' in str(file):
            continue
        print("File:", str(file), "(", count_data+1, "/", size_directory, ")")
        count_data = count_data+1
        signal, samplerate = librosa.core.load(file)
        length_snippets_array = length_snippet * samplerate
        # calculate number of snippets in this file. discard last snippet if too short
        number_of_snippets = len(signal)//length_snippets_array

        # iterate over all snippets in file and calculate stft matrix for each snippet
        i = 0
        for _ in range(number_of_snippets):
            snippet = signal[length_snippets_array*i:length_snippets_array*(i+1)]
            i += 1
            mel_matrix = librosa.feature.melspectrogram(y=snippet, sr=SAMPLERATE,
                    n_mels=NUMBER_OF_MEL_BANDS, fmax=MAX_FREQ_MEL_BAND)
            snippet_list_mel_matrices.append(mel_matrix)
        if CHALLENGE:
            save_as_npy(snippet_list_mel_matrices,
                        "data/classification_test_data/output-challenge-npy-files/" + file.name[:-4])
        else:
            save_as_npy(snippet_list_mel_matrices, "data/classification_test_data/output-labelled-npy-files/" + file.name[:-4])



if __name__ == '__main__':
    create_directory()

    if CHALLENGE:
        # for challenge:
        load_snippets_as_mel_matrices(CLASSIFICATION_DATA_PATH + "/challenge_data", SNIPPET_LENGTH, unknown)

    if not CHALLENGE:
        load_snippets_as_mel_matrices(CLASSIFICATION_DATA_PATH + "/no_leak_eval_data", SNIPPET_LENGTH, no_leak_list)
        load_snippets_as_mel_matrices(CLASSIFICATION_DATA_PATH + "/leak_eval_data", SNIPPET_LENGTH, leak_list)
        print("Files in no_leak_list: ", no_leak_list)
        print("Files in leak_list: ", leak_list)

        with open(CLASSIFICATION_DATA_PATH + '/output-labelled-npy-files/no_leak_list.txt', 'w') as f:
            for item in no_leak_list:
                f.write("%s\n" % item)

        with open(CLASSIFICATION_DATA_PATH + '/output-labelled-npy-files/leak_list.txt', "w") as f:
            for item in leak_list:
                f.write("%s\n" % item)
