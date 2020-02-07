#PATHS
INPUT_PATH = "Material/train_wavs/"
OUTPUT_PATH = "Material/output"
LEAK_PATH = "Material/leak_data_for_testing"
FILTERED_PATH = "Material/train_wavs_filtered/"
SNIPPETS_NAME = "mel_snippets"
LEAK_TEST_DATA_SNIPPETS_NAME = "leak_test_mel_snippets"

#parameters for preprocessing
SNIPPET_LENGTH = 2 #in seconds
SAMPLERATE = 22050
NUMBER_OF_MEL_BANDS = 64



#LOAD mel_snipped in run simplest_autoencoder
LOAD_SNIPPED_FROM_OUTPUT = True

#lead autoencoder_weights_encoder/decoder.h5
LOAD_WEIGHTS = False

#Use Bandpass waves
USE_BANDPASS_FILTERED_WAVS = False

#Preprocessing Wavs with Bandpassfilter
FILTER_WAVS_WITH_BANDPASS = False

#CNN Config
EPOCHS = 2
BATCH_SIZE = 64
SHUFFLE = True