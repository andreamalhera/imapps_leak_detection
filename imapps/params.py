# TODO Refactoring paths to avoid redundancy
#PATHS
PLOT_LEAK_NAME="/leak"
PLOT_NO_LEAK_NAME="/no_leak"
FILTERED_FILES_SUFFIX="_bandpassed.wav"


DATA_HOME="data"
INPUT_PATH = DATA_HOME+"/train_wavs/"
OUTPUT_PATH = DATA_HOME+"/output"
PICS_PATH = DATA_HOME+"/pics/"
WEIGHTS_PATH= DATA_HOME+"/weights/"
LEAK_SNIPPETS_PATH = DATA_HOME + "/leak_data_for_testing"
FILTERED_PATH = DATA_HOME+"/train_wavs_filtered/"
SNIPPETS_PATH = OUTPUT_PATH+"/mel_snippets"
LEAK_TEST_DATA_SNIPPETS_PATH= OUTPUT_PATH+"/leak_test_mel_snippets"
TNSE_PATH=PICS_PATH+"tsne4.png"
PLOT_LEAK_PATH=PICS_PATH+"leak.png"
PLOT_NO_LEAK_PATH=PICS_PATH+"no_leak.png"
AUTOENCODER_WEIGHTS_PATH=WEIGHTS_PATH+"SIMPLE_weights"
ENCODER_WEIGHTS_PATH=WEIGHTS_PATH+"SIMPLE_weights_encoder"
ENCODER_WEIGHTS_PATH_VAE=WEIGHTS_PATH+"VAE_weights_encoder"
CNN_OUTPUT_PATH=OUTPUT_PATH+"/cnn"
MODEL_DATA_FOLDER = "data/model_data"



# Preprocessing
SNIPPET_LENGTH = 2 #in seconds
SAMPLERATE = 22050
NUMBER_OF_MEL_BANDS = 64
MAX_FREQ_MEL_BAND=SAMPLERATE/2.0

# Autoencoders
LOAD_LEAK_SNIPPEDS_FROM_OUTPUT = True
LOAD_SNIPPED_FROM_OUTPUT = True