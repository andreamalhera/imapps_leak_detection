from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import keras.callbacks as t
from keras.utils import plot_model
import audio_test as a
import matplotlib.pyplot as plt


#Autoencoder learns to reconstruct each input sequence
def LTSM(sequence):
    # reshape input into [samples, timesteps, features]
    n_in = len(sequence)
    sequence = sequence.reshape((1, n_in, 1))
    tbCallBack = t.TensorBoard(log_dir='./Graph', histogram_freq=0,
                                             write_graph=True, write_images=True)
    # define model
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_in,1)))
    model.add(RepeatVector(n_in))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    print(model.summary())
    # fit model
    model.fit(sequence, sequence, epochs=100, callbacks = [tbCallBack])

    #GraphVis Error?
    #plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')

    # demonstrate recreation
    yhat = model.predict(sequence, verbose=0)
    print("LTSM Autoencoder try to reconstruct the input (len(outputs) = len(input)): ")
    print(yhat[0,:,0])
    buffer =yhat[0,:,0]
    # Print reconstructed array as Plot
    plt.plot(buffer[5:], '.', ms=0.3, c='r')
    plt.title('Reconstructed Array (1)')
    plt.show()


# Modified LSTM Autoencoder -> instead predict the next step in the sequence.
def LTSM_predict_next_step(seq_in):

    # reshape input into [samples, timesteps, features]
    n_in = len(seq_in)
    seq_in = seq_in.reshape((1, n_in, 1))
    # prepare output sequence
    seq_out = seq_in[:, 1:, :]
    n_out = n_in - 1
    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_in, 1)))
    model.add(RepeatVector(n_out))
    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mse')

    #TODO: GraphViz Error ?
    #plot_model(model, show_shapes=True, to_file='predict_lstm_autoencoder.png')

    # To get some infos about the model:
    #model.summary()

    # fit model
    model.fit(seq_in, seq_out, epochs=100, verbose=0)
    # demonstrate prediction
    yhat = model.predict(seq_in, verbose=0)
    print("LTSM Autoencoder predicts next step (len(outputs) = len(input)-1 ): ")
    print(yhat[0, :, 0])
    #Print reconstructed array as Plot
    plt.plot(yhat[0, :, 0], '.', ms=0.3, c='r')
    plt.title('Reconstructed Array (2)')
    plt.show()




# define input sequence

#seq_in = array([[-0.1],[0.2],[0.3]])
path="Material/duelferstr_sample_1.wav"
seq_in = a.get_audio_data(path, 100)

LTSM(seq_in)
#LTSM_predict_next_step(seq_in)
