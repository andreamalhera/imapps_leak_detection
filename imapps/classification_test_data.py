import cnn_autoencoder
from numpy import loadtxt
from keras.models import load_model
import params
import classification_test_data_preparation
import numpy as np
from scipy.spatial import distance_matrix
import cnn_autoencoder
from keras.models import Model
from keras.layers import Input
import matplotlib.pyplot as plt
import os
import utilities.test_utilities as test_untilities


# TODO: in verschiedene methoden unterteilen, auch so dass leak und no leak verglichen werden kann
# TODO dann cnn trainiertes ding ausprobieren

path = "data/classification_test_data"
name = "x"
weight_file_name = "CNN_weights_e30_dim10_ba64_2019-08-01-08:32:50.h5"
file_names = []


def setup():
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path + "/classification_output"):
        os.mkdir(path + "/classification_output")
    if os.path.exists(path + "/" + name):
        print("name already taken !!!!!!!!!!!!!!!!!!!!!! ")
        exit()


def create_folder_for_output():
    if not os.path.exists(path + "/" + name):
        os.mkdir(path + "/" + name)


def create_folder_for_output_with_position(position):
    if not os.path.exists(path + "/" + name):
        os.mkdir(path + "/" + name)
    if not os.path.exists(path + "/" + name + "/" + str(position) + name):
        os.mkdir(path + "/" + name + "/" + str(position) + name)


def read_leak_list():
    leak_list = list()
    with open(classification_test_data_preparation.folder + "/output-labelled-npy-files/leak_list.txt") as f:
        for line in f:
            line = line[:-1]
            leak_list.append(line)
    # print("Length leak: ", len(leak_list))
    return leak_list


def read_no_leak_list():
    no_leak_list = list()
    with open(classification_test_data_preparation.folder + "/output-labelled-npy-files/no_leak_list.txt") as f:
        for line in f:
            line = line[:-1]
            no_leak_list.append(line)
    # print("Data: ", classification_test_data_preparation.folder + "/" + no_leak_list[0] + ".npy")
    return no_leak_list


def load_autoenc_model():
    print("load ", params.WEIGHTS_PATH + weight_file_name)
    # load model and put weights into model
    model = load_model(params.WEIGHTS_PATH + weight_file_name)
    # summarize model.
    # model.summary()
    return model


# load dataset for first file from list
def load_elem_as_dataset(list, position):
    dataset_npy = np.load(classification_test_data_preparation.folder + "/output-labelled-npy-files/" + list[position] + ".npy") # shape: 450,64,87
    print("Data: ", classification_test_data_preparation.folder + "/" + list[position] + ".npy")
    file_names.append(list[position])
    return dataset_npy


def load_no_leak_dataset(no_leak_list):
    dataset_npy = np.load(classification_test_data_preparation.folder + "/output-labelled-npy-files/" + no_leak_list[0] + ".npy") # shape: 450,64,87
    return dataset_npy


def reshape_dataset_for_input(dataset_npy):
    dataset = dataset_npy.reshape((len(dataset_npy), np.prod(dataset_npy.shape[1:]))) # shape: 450, 5568
    print("original dataset shape: ", dataset_npy.shape)
    return dataset


def reshape_for_cnn_input(dataset_npy):
    dataset = np.reshape(dataset_npy, (-1, dataset_npy.shape[1], dataset_npy.shape[2], 1))
    print("original dataset shape: ", dataset_npy.shape)
    return dataset


def predict_on_dataset(model, dataset):
    # run .npy file through model
    prediction = model.predict(dataset)  # shape: 450, 5568
    # print(prediction)
    # print(prediction.shape)
    return prediction


# reshapes prediction back to mel matrices shape
def reshape_prediction(prediction):
    # pred_npy = np.array
    pred_npy = prediction.reshape(-1, 64, 87)  # prediction.reshape(,64,87)
    # print(pred_npy)
    print("prediction dataset new shape: ", pred_npy.shape)
    return pred_npy


def calculate_euclidean(dataset_npy, pred_npy):
    i = 0
    distance_matrices = []
    for matrix in dataset_npy:
        # distance = distance_matrix(matrix, pred_npy[i]) #.euclidean(matrix, pred_npy[i]) outputs matrix of shape 64*64 ???
        distance = np.subtract(matrix,pred_npy[i])
        distance = np.square(distance)
        # print(matrix)
        # print(prediction)
        distance_matrices.append(distance)
        # print(distance)
        i = i + 1
    # return distance_matrices
    # print("Lengeth of matrix list: " + str(len(distance_matrices)))
    # print("Shape of first element of matrix list: " + str(len(distance_matrices[0])))
    # print(" First element: ", distance_matrices[0])
    # print("shape of first element: ", len(distance_matrices[0][0]))

    # for every matrix transpose and calculate sum and sqrt --> euclidean dist
    re_matrix = []
    for element in distance_matrices:
        # switch from melbands,frames to frames,melbands (or other way round?? not sure yet)
        np.transpose(element)
        array_list = []
        for array in element:
            euclid = np.sum(array)
            euclid = np.sqrt(euclid)
            array_list.append(euclid)
        re_matrix.append(array_list)

    # print(re_matrix)
    # print("re_matrix ", len(re_matrix))
    # print("len re_matrix ", len(re_matrix[0]))
    # print("len re matrix elem", re_matrix[0][0]))
    return re_matrix


# get final reconstruction error
def mean_for_every_snippet(re_matrix):
    # reconstruction error mean for every 2 second snippet
    re_final_list = []
    for element in re_matrix:
        sum_of_all_re = np.sum(element)
        mean = sum_of_all_re/87  # sum/number of frames
        re_final_list.append(mean)

    print(re_final_list)
    return re_final_list


def get_cnn_model():
    # this is our input placeholder. Shape needed for first layer.
    input_shape = (64,87,1)
    input_training = Input(shape=input_shape)

    # Encoded layers
    encoded = cnn_autoencoder.encoding_layers(input_training, 2, 1)
    # Decoded layers
    decoded = cnn_autoencoder.decoding_layers(encoded, 2, 1)

    autoencoder = Model(input_training, decoded)
    autoencoder.compile(optimizer='adadelta', loss=cnn_autoencoder.LOSS_CNN)
    return autoencoder


def sum_up_error_for_whole_file(errors):
    sum = np.sum(errors)
    return sum


def plot_REs_as_lines(means_RE_leak, means_RE_no_leak, means_RE_challenge, position):
    x1 = np.arange(len(means_RE_leak))
    x2 = np.arange(len(means_RE_no_leak))
    x3 = np.arange(len(means_RE_challenge))

    y1 = means_RE_leak
    y2 = means_RE_no_leak
    y3 = means_RE_challenge

    plt.plot(x1, y1, c="r", label="leak")
    plt.plot(x2, y2, c="g", label="no_leak")
    plt.plot(x3, y3, c="b", label="challenge")
    plt.legend()
    if position == "None":
        plt.savefig("data/classification_test_data/" + name + "/" + name + "_RE.png")
    else:
        plt.savefig("data/classification_test_data/" + name + "/"
                    + str(position) + name + "/" + name + "_RE.png")
    plt.show()


def plot_REs_as_lines_details(means_RE_leak, means_RE_no_leak, means_RE_challenge, y_limit, position):
    x1 = np.arange(len(means_RE_leak))
    x2 = np.arange(len(means_RE_no_leak))
    x3 = np.arange(len(means_RE_challenge))

    y1 = means_RE_leak
    y2 = means_RE_no_leak
    y3 = means_RE_challenge

    plt.plot(x1, y1, c="r", label="leak")
    plt.plot(x2, y2, c="g", label="no_leak")
    plt.plot(x3, y3, c="b", label="challenge")
    plt.ylim(0, y_limit)
    plt.legend()

    if position == "None":
        plt.savefig("data/classification_test_data/" + name + "/" + name + "_RE_details.png")
    else:
        plt.savefig("data/classification_test_data/" + name + "/"
                    + str(position) + name + "/" + name + "_RE_details.png")
    plt.show()


def save_details(model, means_RE_leak, sum_leak, means_RE_no_leak, sum_no_leak, position):
    if position == "None":
        f = open("data/classification_test_data/" + name + "/" + name + "info.txt", "w+")
    else:
        f = open("data/classification_test_data/"+ name + "/"
                 + str(position) + name + "/" + name + "info.txt", "w+")

    f.write("Name: " + name + "\n")
    f.write("Weights: " + weight_file_name + "\n")
    for element in file_names:
        f.write(element)
        f.write('\n')
    f.write("Means RE leak: " + str(means_RE_leak) + "\n")
    f.write("Means RE no leak: " + str(means_RE_no_leak) + "\n")
    f.write("\n")
    f.write("Sum RE leak:" + str(sum_leak) + "\n")
    f.write("Sum RE no leak: " + str(sum_no_leak) + "\n")
    f.write("\n")

    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.close()


def classify(model_weights_name):

    trained_weights_path = params.WEIGHTS_PATH + model_weights_name
    model = load_model(trained_weights_path)

    leak_list = read_leak_list()
    no_leak_list = read_no_leak_list()

    leak_label = 1
    no_leak_label = 0

    filename_error_label_list = []

    for i in range(len(leak_list)):
        data_leak_npy = load_elem_as_dataset(leak_list, i)

        if "CNN" in model_weights_name or "cnn" in model_weights_name or "Cnn" in model_weights_name:
            data_leak = reshape_for_cnn_input(data_leak_npy)
        else:
            data_leak = reshape_dataset_for_input(data_leak_npy)

        prediction_leak = predict_on_dataset(model, data_leak)
        pred_leak_npy = reshape_prediction(prediction_leak)

        # get reconstruction error
        euclidean_matrices_leak = calculate_euclidean(data_leak_npy, pred_leak_npy)
        means_RE_leak = mean_for_every_snippet(euclidean_matrices_leak)
        sum_RE_leak = sum_up_error_for_whole_file(means_RE_leak)

        filename_error_label_list.append([leak_list[i], sum_RE_leak, leak_label])

    for i in range(len(no_leak_list)):
        data_leak_npy = load_elem_as_dataset(leak_list, i)

        if "CNN" in model_weights_name or "cnn" in model_weights_name or "Cnn" in model_weights_name:
            data_leak = reshape_for_cnn_input(data_leak_npy)
        else:
            data_leak = reshape_dataset_for_input(data_leak_npy)

        prediction_leak = predict_on_dataset(model, data_leak)
        pred_leak_npy = reshape_prediction(prediction_leak)

        # get reconstruction error
        euclidean_matrices_leak = calculate_euclidean(data_leak_npy, pred_leak_npy)
        means_RE_leak = mean_for_every_snippet(euclidean_matrices_leak)
        sum_RE_leak = sum_up_error_for_whole_file(means_RE_leak)

        filename_error_label_list.append([leak_list[i], sum_RE_leak, no_leak_label])

    print("CLASSIFICATION RESULTS: ", filename_error_label_list)
    return filename_error_label_list


def get_TSNE():
    cnn = False
    encoder = load_model(params.WEIGHTS_PATH + "SIMPLE_weights_encoder_e30_dim10_ba64_2019-08-01-10:28:10.h5")

    leak_list = read_leak_list()
    no_leak_list = read_no_leak_list()

    dataset_leak_npy = load_elem_as_dataset(leak_list, 3)
    dataset_no_leak_npy = load_elem_as_dataset(no_leak_list, 3)

    if cnn:
        dataset_leak = reshape_for_cnn_input(dataset_leak_npy)
        dataset_no_leak = reshape_for_cnn_input(dataset_no_leak_npy)
    else:
        dataset_leak = reshape_dataset_for_input(dataset_leak_npy)
        dataset_no_leak = reshape_dataset_for_input(dataset_no_leak_npy)

    prediction_vector_leak = encoder.predict(dataset_leak)
    prediction_vector_leak = prediction_vector_leak.reshape(-1, 5568)

    prediction_vector_no_leak = encoder.predict(dataset_no_leak)
    prediction_vector_no_leak = prediction_vector_no_leak.reshape(-1, 5568)

    test_untilities.tsne_presentation_of_vectors(prediction_vector_no_leak, prediction_vector_leak)



if __name__ == '__main__':
    get_TSNE()
    ## TODO: every time you start a new classification set name here
    ## ONLY SET NAME HERE, NEVER CHANGE OTHER NAME VARIABLES!!!
    #name = "test_Challenge_test3.2"
    ## weight_file_name = "cnn_epochs250_dim100_batch64_initial_architecture_weights.h5"
    #weight_file_name = "CNN_weights_e30_dim10_ba64_2019-08-01-08:32:50.h5"
    #cnn = True
    #
    #setup()
    #
    #model = load_autoenc_model()
    #
    ## preparation
    #leak_list = read_leak_list()
    #no_leak_list = read_no_leak_list()
    #
    #num_of_files = min(len(leak_list), len(no_leak_list))
    #
    #for i in range(num_of_files):
    #    create_folder_for_output_with_position(i)
    #    data_leak_npy = load_elem_as_dataset(leak_list, i)
    #    data_no_leak_npy = load_elem_as_dataset(no_leak_list, i)
    #
    #    if cnn:
    #        data_leak = reshape_for_cnn_input(data_leak_npy)
    #        data_no_leak = reshape_for_cnn_input(data_no_leak_npy)
    #    else:
    #        data_leak = reshape_dataset_for_input(data_leak_npy)
    #        data_no_leak = reshape_dataset_for_input(data_no_leak_npy)
    #
    #    prediction_leak = predict_on_dataset(model, data_leak)
    #    pred_leak_npy = reshape_prediction(prediction_leak)
    #
    #    prediction_no_leak = predict_on_dataset(model, data_no_leak)
    #    pred_no_leak_npy = reshape_prediction(prediction_no_leak)
    #
    #    # get reconstruction error
    #    # --> jede ursprüngliche npy matrix mit prediction vergleichen
    #    euclidean_matrices_leak = calculate_euclidean(data_leak_npy, pred_leak_npy)
    #    means_RE_leak = mean_for_every_snippet(euclidean_matrices_leak)
    #
    #    euclidean_matrices_no_leak = calculate_euclidean(data_no_leak_npy, pred_no_leak_npy)
    #    means_RE_no_leak = mean_for_every_snippet(euclidean_matrices_no_leak)
    #
    #    plot_REs_as_lines(means_RE_leak, means_RE_no_leak, i)
    #    plot_REs_as_lines_details(means_RE_leak, means_RE_no_leak, 0.25, i)
    #
    #    sum_leak = sum_up_error_for_whole_file(means_RE_leak)
    #    sum_no_leak = sum_up_error_for_whole_file(means_RE_no_leak)
    #
    #    print("LEAK:")
    #    print(means_RE_leak)
    #    print(sum_up_error_for_whole_file(means_RE_leak))
    #    print("###########################")
    #    print("NO_LEAK:")
    #    print(means_RE_no_leak)
    #    print(sum_up_error_for_whole_file(means_RE_no_leak))
    #
    #    save_details(model, means_RE_leak, sum_leak, means_RE_no_leak, sum_no_leak, i)

    ########### OLD: just first file, ^NEW: all 8 (or so) test files

    #create_folder_for_output()
    #
    #dataset_leak_npy = load_elem_as_dataset(leak_list, 0)
    #dataset_no_leak_npy = load_elem_as_dataset(no_leak_list, 0)
    #
    #dataset_challenge_npy = np.load(
    #    classification_test_data_preparation.folder + "/output-challenge-npy-files/mel_snippets_Other_Dif.npy")
    #
    #dataset_challenge_npy = dataset_challenge_npy[:450]
    #
    #print("CHALLENGE", len(dataset_challenge_npy))
    #
    #if cnn:
    #   dataset_leak = reshape_for_cnn_input(dataset_leak_npy)
    #   dataset_challenge = reshape_for_cnn_input(dataset_challenge_npy)
    #   dataset_no_leak = reshape_for_cnn_input(dataset_no_leak_npy)
    #else:
    #   dataset_leak = reshape_dataset_for_input(dataset_leak_npy)
    #   dataset_challenge = reshape_dataset_for_input(dataset_challenge_npy)
    #   dataset_no_leak = reshape_dataset_for_input(dataset_no_leak_npy)
    #
    ## use for file where only weights not model of cnn are saved:
    ## model = get_cnn_model()
    ## model.load_weights(params.WEIGHTS_PATH + weight_file_name)
    #
    #prediction_leak = predict_on_dataset(model, dataset_leak)
    #pred_leak_npy = reshape_prediction(prediction_leak)
    #
    #prediction_no_leak = predict_on_dataset(model, dataset_no_leak)
    #pred_no_leak_npy = reshape_prediction(prediction_no_leak)
    #
    #prediction_challenge = predict_on_dataset(model, dataset_challenge)
    #pred_challenge_npy = reshape_prediction(prediction_challenge)
    #
    ## get reconstruction error
    ## --> jede ursprüngliche npy matrix mit predition vergleichen
    #euclidean_matrices_leak = calculate_euclidean(dataset_leak_npy, pred_leak_npy)
    #means_RE_leak = mean_for_every_snippet(euclidean_matrices_leak)
    #
    #euclidean_matrices_challenge = calculate_euclidean(dataset_challenge_npy, pred_challenge_npy)
    #means_RE_challenge = mean_for_every_snippet(euclidean_matrices_challenge)
    #
    #euclidean_matrices_no_leak = calculate_euclidean(dataset_no_leak_npy, pred_no_leak_npy)
    #means_RE_no_leak = mean_for_every_snippet(euclidean_matrices_no_leak)
    #
    #print("LEAK:")
    #print(means_RE_leak)
    #print(sum_up_error_for_whole_file(means_RE_leak))
    #print("###########################")
    #print("NO_LEAK:")
    #print(means_RE_no_leak)
    #print(sum_up_error_for_whole_file(means_RE_no_leak))
    #
    #plot_REs_as_lines(means_RE_leak, means_RE_no_leak, means_RE_challenge, "None")
    #plot_REs_as_lines_details(means_RE_leak, means_RE_no_leak, means_RE_challenge, 0.25, "None")
    #
    #sum_leak = sum_up_error_for_whole_file(means_RE_leak)
    #sum_no_leak = sum_up_error_for_whole_file(means_RE_no_leak)
    #
    #save_details(model, means_RE_leak, sum_leak, means_RE_no_leak, sum_no_leak, "None")
    #
    ## TODO: get percentage of high reconstruction error ??
    #
    ## old, might need later
    ## preprocessed_snippets
    ## for i in range(len(leak_list)):
    ##    print("Current file: ", leak_list[i])
    ##    file = np.load(classification_test_data_preparation.folder + "/output-labelled-npy-files/" + leak_list[i] + ".npy")
    ##    file = file.reshape((len(file), np.prod(file.shape[1:])))
    ##    prediction = model.predict(file)

