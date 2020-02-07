import utilities.test_utilities as u
import classification_test_data as c
import numpy as np



if __name__ == '__main__':
    # TODO: every time you start a new classification set name here
    # ONLY SET NAME HERE, NEVER CHANGE OTHER NAME VARIABLES!!!
    name = "test_cnn_10_epochs"
    # weight_file_name = "cnn_epochs250_dim100_batch64_initial_architecture_weights.h5"
    weight_file_name = "CNN_weights_e30_dim10_ba64_2019-08-01-08:32:50.h5"
    cnn = True

    # preparation
    leak_list = c.read_leak_list()
    no_leak_list = c.read_no_leak_list()
    index = 1
    dataset_leak_npy = c.load_elem_as_dataset(leak_list,0)
    dataset_no_leak_npy = c.load_elem_as_dataset(no_leak_list,0)


    dataset_challenge_npy = np.load(
        c.classification_test_data_preparation.folder + "/output-challenge-npy-files/mel_snippets_Other_Dif.npy")

    dataset_challenge_npy = dataset_challenge_npy[450:900]

    if cnn:
        dataset_leak = c.reshape_for_cnn_input(dataset_leak_npy)
        dataset_no_leak = c.reshape_for_cnn_input(dataset_no_leak_npy)
        dataset_challenge = c.reshape_for_cnn_input(dataset_challenge_npy)
    else:
        dataset_leak = c.reshape_dataset_for_input(dataset_leak_npy)
        dataset_challenge = c.reshape_dataset_for_input(dataset_challenge_npy)
        dataset_no_leak = c.reshape_dataset_for_input(dataset_no_leak_npy)

    # model = load_autoenc_model("cnn_epochs250_dim100_batch64_initial_architecture_weights.h5")
    model = c.load_autoenc_model()

    # use for file where only weights not model of cnn are saved:
    # model = get_cnn_model()
    # model.load_weights(params.WEIGHTS_PATH + weight_file_name)

    prediction_leak = c.predict_on_dataset(model, dataset_leak)
    #pred_leak_npy = c.reshape_prediction(prediction_leak)

    prediction_no_leak = c.predict_on_dataset(model, dataset_no_leak)
    #pred_no_leak_npy = c.reshape_prediction(prediction_no_leak)

    prediction_diff = c.predict_on_dataset(model, dataset_challenge)

    start = 0
    u.plot_encoded_decoded(dataset_leak,prediction_leak,"data/classification_test_data/test_Challenge_test3.2/"+weight_file_name[:-3] + "_leak.png")
    u.plot_encoded_decoded(dataset_no_leak,prediction_no_leak,"data/classification_test_data/test_Challenge_test3.2/"+weight_file_name[:-3] + "_no_leak.png")
    u.plot_encoded_decoded(dataset_challenge,prediction_diff,"data/classification_test_data/test_Challenge_test3.2/"+weight_file_name[:-3] + "_challenge.png")


