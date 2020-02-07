import preprocessing
import params
import os
import numpy as np


def create_directories():
    if not os.path.exists(params.LEAK_PATH):
        os.mkdir(params.LEAK_PATH)

def run_preprocessing():
    input_path = params.LEAK_PATH
    snippets = preprocessing.load_snippets_as_mel_matrices(input_path, params.SNIPPET_LENGTH)
    snippets_np = np.asarray(snippets)
    print("INFO: Shape of preprocessed " + params.LEAK_TEST_DATA_SNIPPETS_NAME+": ", snippets_np.shape)

    return snippets_np


if __name__ == '__main__':
    create_directories()

    SNIPPETS_PATH = params.OUTPUT_PATH+"/"+params.LEAK_TEST_DATA_SNIPPETS_NAME

    snippets = run_preprocessing()

    np.save(SNIPPETS_PATH, snippets)
    print("INFO: Snippets saved in " + SNIPPETS_PATH + " as .npy")


