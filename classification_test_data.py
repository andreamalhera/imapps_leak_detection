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

# TODO: in verschiedene methoden unterteilen, auch so dass leak und no leak verglichen werden kann
# TODO dann cnn trainiertes ding ausprobieren

WEIGHTS_PATH=params.WEIGHTS_PATH

def read_leak_list():
  leak_list = list()
  with open(classification_test_data_preparation.CLASSIFICATION_DATA_PATH + "/output-labelled-npy-files/leak_list.txt") as f:
    for line in f:
      line = line[:-1]
      leak_list.append(line)
  # print("Lenght leak: ", len(leak_list))
  return leak_list


def read_no_leak_list():
  no_leak_list = list()
  with open(classification_test_data_preparation.CLASSIFICATION_DATA_PATH + "/output-labelled-npy-files/no_leak_list.txt") as f:
    for line in f:
      line = line[:-1]
      no_leak_list.append(line)
  #print("Data: ", classification_test_data_preparation.folder + "/" + no_leak_list[0] + ".npy")
  return no_leak_list


def load_autoenc_model(weights_name):
  print(WEIGHTS_PATH + weights_name)
  # load model and put weights into model
  model = load_model(WEIGHTS_PATH + weights_name)
  # summarize model.
  #model.summary()
  return model


# load dataset for first file from leak list
def load_first_elem_as_dataset(list):
  dataset_npy = np.load(classification_test_data_preparation.CLASSIFICATION_DATA_PATH + "/output-labelled-npy-files/" + list[0] + ".npy") # shape: 450,64,87
  print("Data: ", classification_test_data_preparation.CLASSIFICATION_DATA_PATH + "/" + list[0] + ".npy")
  return dataset_npy


def load_no_leak_dataset(no_leak_list):
  dataset_npy = np.load(classification_test_data_preparation.CLASSIFICATION_DATA_PATH + "/output-labelled-npy-files/" + leak_list[0] + ".npy") # shape: 450,64,87


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

#reshapes prediction back to mel matrices shape
def reshape_prediction(prediction):
  #pred_npy = np.array
  pred_npy = prediction.reshape(-1,64,87) #prediction.reshape(,64,87)
  #print(pred_npy)
  print("prediction dataset new shape: ",pred_npy.shape)
  return pred_npy


def calculate_euclidean(dataset_npy, pred_npy):
  i = 0
  distance_matrices = []
  for matrix in dataset_npy:
    #distance = distance_matrix(matrix, pred_npy[i]) #.euclidean(matrix, pred_npy[i]) outputs matrix of shape 64*64 ???
    distance = np.subtract(matrix,pred_npy[i])
    distance = np.square(distance)
    #print(matrix)
    #print(prediction)
    distance_matrices.append(distance)
    #print(distance)
    i = i + 1
  #return distance_matrices
  #print("Lengeth of matrix list: " + str(len(distance_matrices)))
  #print("Shape of first element of matrix list: " + str(len(distance_matrices[0])))
  #print(" First element: ", distance_matrices[0])
  #print("shape of first element: ", len(distance_matrices[0][0]))

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

  #print(re_matrix)
  #print("re_matrix ", len(re_matrix))
  #print("len re_matrix ", len(re_matrix[0]))
  # print("len re matrix elem", re_matrix[0][0]))
  return re_matrix


# get final reconstruction error
def mean_for_every_snippet(re_matrix):
  # reconstruction error mean for every 2 second snippet
  re_final_list = []
  for element in re_matrix:
    sum_of_all_re = np.sum(element)
    mean = sum_of_all_re/87 #sum/number of frames
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


if __name__ == '__main__':
  # preparation
  leak_list = read_leak_list()
  no_leak_list = read_no_leak_list()

  dataset_leak_npy = load_first_elem_as_dataset(leak_list)
  #dataset_leak = reshape_dataset_for_input(dataset_leak_npy)
  dataset_leak = reshape_for_cnn_input(dataset_leak_npy)

  dataset_no_leak_npy = load_first_elem_as_dataset(no_leak_list)
  #dataset_no_leak = reshape_dataset_for_input(dataset_no_leak_npy)
  dataset_no_leak = reshape_for_cnn_input(dataset_no_leak_npy)

  #model = load_autoenc_model("cnn_epochs250_dim100_batch64_initial_architecture_weights.h5")
  #model = load_autoenc_model("autoencoder_weights.h5")
  model = get_cnn_model()
  model.load_weights("data/weights/cnn_epochs250_dim100_batch64_initial_architecture_weights.h5")

  prediction_leak = predict_on_dataset(model, dataset_leak)
  pred_leak_npy = reshape_prediction(prediction_leak)

  prediction_no_leak = predict_on_dataset(model, dataset_no_leak)
  pred_no_leak_npy = reshape_prediction(prediction_no_leak)


  # get reconstruction error
  # --> jede ursprüngliche npy matrix mit predition vergleichen
  euclidean_matrices_leak = calculate_euclidean(dataset_leak_npy, pred_leak_npy)
  means_RE_leak = mean_for_every_snippet(euclidean_matrices_leak)

  euclidean_matrices_no_leak = calculate_euclidean(dataset_no_leak_npy, pred_no_leak_npy)
  means_RE_no_leak = mean_for_every_snippet(euclidean_matrices_no_leak)

  print("LEAK:")
  print(means_RE_leak)
  print(sum_up_error_for_whole_file(means_RE_leak))
  print("###########################")
  print("NO_LEAK:")
  print(means_RE_no_leak)
  print(sum_up_error_for_whole_file(means_RE_no_leak))


  # TODO: get percentage of high reconstruction error ??



  # old, might need later
  # preprocessed_snippets
  #for i in range(len(leak_list)):
  #    print("Current file: ", leak_list[i])
  #    file = np.load(classification_test_data_preparation.folder + "/output-labelled-npy-files/" + leak_list[i] + ".npy")
  #    file = file.reshape((len(file), np.prod(file.shape[1:])))
  #    prediction = model.predict(file)

