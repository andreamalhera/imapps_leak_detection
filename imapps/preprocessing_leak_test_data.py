import preprocessing
import params
import os
import numpy as np

LEAK_SNIPPETS_PATH=params.LEAK_SNIPPETS_PATH
SNIPPETS_PATH=params.SNIPPETS_PATH
SNIPPET_LENGTH=params.SNIPPET_LENGTH
LEAK_TEST_DATA_SNIPPETS_PATH=params.LEAK_TEST_DATA_SNIPPETS_PATH
LOAD_LEAK_SNIPPEDS_FROM_OUTPUT=params.LOAD_LEAK_SNIPPEDS_FROM_OUTPUT

def create_directories():
    if not os.path.exists(LEAK_SNIPPETS_PATH):
        os.mkdir(LEAK_SNIPPETS_PATH)


def run_preprocessing():
    snippets = preprocessing.load_snippets_as_mel_matrices(LEAK_SNIPPETS_PATH, SNIPPET_LENGTH)
    snippets_np = np.asarray(snippets)
    print("INFO: Shape of preprocessed: ", snippets_np.shape)

    if LOAD_LEAK_SNIPPEDS_FROM_OUTPUT:
        np.save(SNIPPETS_PATH, snippets)
        print("INFO: Snippets saved in " + LEAK_TEST_DATA_SNIPPETS_PATH + ".npy")

    return snippets_np


if __name__ == '__main__':
    create_directories()
    snippets = run_preprocessing()

    np.save(LEAK_TEST_DATA_SNIPPETS_PATH, snippets)
    print("INFO: Snippets saved in " + LEAK_TEST_DATA_SNIPPETS_PATH + ".npy")
