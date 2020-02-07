import librosa
import librosa.display
import os
import numpy as np
import params
from scipy.signal import butter,lfilter

sr = params.SAMPLERATE
length_snippet_sec = params.SNIPPET_LENGTH
number_of_mel_bands = params.NUMBER_OF_MEL_BANDS
#                       as proposed by librosa: samplerate/2
max_freq_of_mel_bands = sr / 2.0


def create_directories():
    if not os.path.exists("Material"):
        os.mkdir("Material")
    if not os.path.exists(params.INPUT_PATH):
        os.mkdir(params.INPUT_PATH)
    if not os.path.exists(params.OUTPUT_PATH):
        os.mkdir(params.OUTPUT_PATH)
    if not os.path.exists(params.FILTERED_PATH):
        os.mkdir(params.FILTERED_PATH)

def Butterworth_filter(signal: [int], samplerate: int, filtertype='high', *cutoff: (int,), order=5):
    print(cutoff, filtertype)
    assert samplerate > 0, "Invalid Samplingrate!!"
    assert len(signal) > 0, "Empty Signal-Array!!"
    assert len(cutoff) < 3, "Filter not yet defined!!"

    nyq = 0.5 * samplerate  # max darstellbare Hz
    normal_cutoff = list((val / nyq for val in cutoff))  # wieviel % man wegnehmen möchte
    numerator, denominator = butter(order, normal_cutoff, filtertype, analog=False)
    filtered_data = lfilter(numerator, denominator, signal)
    return filtered_data


def bandpass_over_files(path_to_input_files, lower, upper):
    for file in os.scandir(path_to_input_files):
        if 'Store' in str(file):
            continue
        signal, samplerate = librosa.core.load(file)
        band_pass_signal = Butterworth_filter(signal, samplerate, 'band', lower, upper)
        librosa.output.write_wav(params.FILTERED_PATH + str(file.name)[:-4] + "_bandpassed.wav", band_pass_signal, sr, norm=False)


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
    for file in os.scandir(path_to_files):
        if 'Store' in str(file):
            continue
        signal, samplerate = librosa.core.load(file)
        length_snippets_array = length_snippet * samplerate
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

# TODO: Add error handling for non-existing INPUT_PATH directory and non-existing .wav-files
def run_preprocessing():
    # STFT Snippets
    #snippets = load_snippets_as_stft_matrices(dir_input, length_snippet_sec)

    if params.USE_BANDPASS_FILTERED_WAVS == True:
        input_path = params.FILTERED_PATH
    else:
        input_path = params.INPUT_PATH

    snippets = load_snippets_as_mel_matrices(input_path, length_snippet_sec)

    # Output as numpy array
    snippets_np = np.asarray(snippets)
    print("INFO: Shape of preprocessed " + params.SNIPPETS_NAME+": ", snippets_np.shape)

    return snippets_np

if __name__ == '__main__':
    create_directories()
    #TODO: Move SNIPPETS_PATH to params (also used by simplest_encoder.py)
    SNIPPETS_PATH = params.OUTPUT_PATH+"/"+params.SNIPPETS_NAME

    #if True wavs we be filtered before snipping
    if params.FILTER_WAVS_WITH_BANDPASS:
        bandpass_over_files(params.INPUT_PATH, 300, 1000)
        print("INFO: Bandpassed .wav files saved in " + params.FILTERED_PATH)

    snippets = run_preprocessing()

    # write mel snippet list to .npy file
    np.save(SNIPPETS_PATH, snippets)
    print("INFO: Snippets saved in "+SNIPPETS_PATH+".npy")


    # to print pretty
    # np.set_printoptions(suppress=True, threshold = sys.maxsize)
    # x = np.asmatrix(mel_snippets[0])
    # print(x)


