import params
import numpy as np
import os


"""
This script splits all data with no leaks (WITHOUT the .wav-files in the black list!) into training data and test data.
The training data should be used to train the autoencoder models.
The test data should be used to get a clue of how well the autoencoder model reconstructs the input it gets.
"""


MODEL_DATA_PATH = "data/model_data"
TEST_PERCENTAGE = 0.2
SNIPPETS_PATH=params.SNIPPETS_PATH

def create_directory():
    if not os.path.exists(MODEL_DATA_PATH):
        os.mkdir(MODEL_DATA_PATH)


def load_initial_data():
    # load data from .npy-file

    data = np.load(SNIPPETS_PATH + ".npy")

    # TODO:print("No data found. Run preprocessing first")
    return data


def split_data(dataset):
    # split in train and test data
    x_train = dataset[:int(dataset.shape[0] * (1 - TEST_PERCENTAGE)), :]
    print("INFO: Use data from position 0 to position ", int(dataset.shape[0] * (1 - TEST_PERCENTAGE)),
          " as training data.")
    x_test = dataset[int(dataset.shape[0] * (1 - TEST_PERCENTAGE)):, :]
    print("INFO: Use data from position ", int(dataset.shape[0] * (1 - TEST_PERCENTAGE)), " to position ",
          dataset.shape[0], " as test data.")
    return x_train, x_test


def save_as_npy(dataset, path):
    # write mel snippet list to .npy file
    np.save(path + ".npy", dataset)
    print("INFO: Snippets saved in " + path + ".npy")
    return


if __name__ == '__main__':
    create_directory()
    whole_dataset = load_initial_data()
    train_data, test_data = split_data(whole_dataset)
    save_as_npy(train_data, MODEL_DATA_PATH + "/train_data")
    save_as_npy(test_data, MODEL_DATA_PATH + "/test_data")
