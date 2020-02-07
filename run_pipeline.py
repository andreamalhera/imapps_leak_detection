from simple_autoencoder import run_simple_autoencoder
from cnn_autoencoder import run_cnn_autoencoder

def is_everything_set():
    '''
    Check if every needed directory is already there. This method could also be model specific
    :return: Bool, true if everythin is set
    '''
    return True

def run_all_models():
    run_simple_autoencoder()
    run_cnn_autoencoder()

# TODO Call preprocessing
def run_all_preprocessing():
    return

def test_all_models():
    exit()

if __name__ == '__main__':
    run_all_preprocessing()
    run_all_models()
