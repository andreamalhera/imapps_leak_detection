import librosa
import librosa.display
import os
import numpy as np
import params
from scipy.signal import butter,lfilter

USE_BANDPASS_FILTERED_WAVS = False
FILTER_WAVS_WITH_BANDPASS = False
NYQ = 0.5*params.SAMPLERATE # max darstellbare Hz
BANDPASS_LOWER_BOUND=300
BANDPASS_UPPER_BOUND=1000
MIN_SAMPLERATE=0
MAX_CUTOFF=3

SAMPLERATE = params.SAMPLERATE
LENGTH_SNIPPET_SEC = params.SNIPPET_LENGTH
NUMBER_OF_MEL_BANDS = params.NUMBER_OF_MEL_BANDS # as proposed by librosa: samplerate/2
MAX_FREQ_MEL_BAND =params.MAX_FREQ_MEL_BAND
SNIPPETS_PATH=params.SNIPPETS_PATH
FILTERED_FILES_SUFFIX=params.FILTERED_FILES_SUFFIX
FILTERED_PATH=params.FILTERED_PATH
INPUT_PATH=params.INPUT_PATH
OUTPUT_PATH=params.OUTPUT_PATH
LOAD_SNIPPED_FROM_OUTPUT=params.LOAD_SNIPPED_FROM_OUTPUT
DATA_HOME=params.DATA_HOME

def create_directories():
    # preprocessing
    if not os.path.exists(DATA_HOME):
        os.mkdir(DATA_HOME)
    if not os.path.exists(INPUT_PATH):
        os.mkdir(INPUT_PATH)
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    if not os.path.exists(FILTERED_PATH):
        os.mkdir(FILTERED_PATH)

def butterworth_filter(signal: [int], samplerate: int, filtertype='high', *cutoff: (int,), order=5):
    print(cutoff, filtertype)
    assert signal.any(), "Empty Signal-Array!!"
    assert samplerate > MIN_SAMPLERATE, "Invalid Samplingrate!!"
    assert len(cutoff) < MAX_CUTOFF, "Filter not yet defined!!"

    normal_cutoff = list((val / NYQ for val in cutoff))  # wieviel % man wegnehmen möchte
    numerator, denominator = butter(order, normal_cutoff, filtertype, analog=False)
    filtered_data = lfilter(numerator, denominator, signal)
    return filtered_data


def bandpass_over_files(path_to_input_files, lower, upper):
    for file in os.scandir(path_to_input_files):
        if 'Store' in str(file):
            continue
        signal, samplerate = librosa.core.load(file)
        band_pass_signal = butterworth_filter(signal, samplerate, 'band', lower, upper)
        librosa.output.write_wav(FILTERED_PATH + str(file.name)[:-4] + FILTERED_FILES_SUFFIX,
                                 band_pass_signal, SAMPLERATE, norm=False)


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
        length_snippets_array = length_snippet * samplerate

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
    count_data = 0
    size_directory = 0
    for file in os.scandir(path_to_files):
        size_directory = size_directory+1
    for file in os.scandir(path_to_files):
        if 'Store' in str(file):
            continue
        print("File:", str(file), "(", count_data+1, "/", size_directory,")")
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
            mel_matrix = librosa.feature.melspectrogram(y=snippet, sr=samplerate,
                                                        n_mels=NUMBER_OF_MEL_BANDS, fmax=MAX_FREQ_MEL_BAND)
            snippet_list_mel_matrices.append(mel_matrix)

    return snippet_list_mel_matrices

# TODO: Add error handling for non-existing INPUT_PATH directory and non-existing .wav-files
def run_preprocessing():
    #if True wavs we be filtered before snipping
    if FILTER_WAVS_WITH_BANDPASS:
        bandpass_over_files(INPUT_PATH, BANDPASS_LOWER_BOUND,
                BANDPASS_UPPER_BOUND)
        print("INFO: Bandpassed .wav files saved in " + FILTERED_PATH)

    # STFT Snippets
    #snippets = load_snippets_as_stft_matrices(dir_input, length_snippet_sec)

    if USE_BANDPASS_FILTERED_WAVS:
        input_path = FILTERED_PATH
    else:
        input_path = INPUT_PATH

    snippets = load_snippets_as_mel_matrices(input_path, LENGTH_SNIPPET_SEC)

    # Output as numpy array
    snippets_np = np.asarray(snippets)
    print("INFO: Shape of preprocessed " + SNIPPETS_PATH+": ", snippets_np.shape)

    # write mel snippet list to .npy file
    if LOAD_SNIPPED_FROM_OUTPUT:
        np.save(SNIPPETS_PATH, snippets)
        print("INFO: Snippets saved in "+SNIPPETS_PATH+".npy")

    return snippets_np

if __name__ == '__main__':
    create_directories()

    snippets = run_preprocessing()

    # write mel snippet list to .npy file
    #np.save(SNIPPETS_PATH, snippets)
    #print("INFO: Snippets saved in "+SNIPPETS_PATH+".npy")


    # to print pretty
    # np.set_printoptions(suppress=True, threshold = sys.maxsize)
    # x = np.asmatrix(mel_snippets[0])
    # print(x)


