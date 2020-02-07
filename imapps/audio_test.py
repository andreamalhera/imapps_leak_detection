import matplotlib.pyplot as plt
import librosa
import numpy as np
import wave
import sys
import librosa.display
import sklearn
from sklearn.preprocessing import MinMaxScaler

path="data/messestelle1.wav"

def plot_audio(filename):

    spf = wave.open(filename,'r')

    #Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')


    #If Stereo
    if spf.getnchannels() == 2:
        print('Just mono files')
        sys.exit(0)

    plt.figure(2)
    plt.title('Signal Wave...')
    plt.plot(signal)
    plt.show()
plot_audio(path)

def spectrogram(path):
    y, sr = librosa.load(path)
    plt.figure(figsize=(24, 16))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.subplot(4, 2, 1)
    librosa.display.specshow(D, y_axis='linear', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
    plt.show()

#read audiofile with target sampling rate
def get_audio_data(filename, target_sr):
    audio, _ = librosa.load(filename, sr=target_sr, mono=True)
    # mono=True = 1, else = 2
    audio = audio.reshape(-1, 1)

    # show -> print Plot
    #audio_norm = sklearn.preprocessing.normalize(audio)

  #  audio_norm = sklearn.preprocessing.scale(audio)
    scaler = MinMaxScaler()
   # scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1), copy=True)
    audio_norm = scaler.fit(audio)
    audio_norm = audio_norm.transform(audio)
    print(audio_norm)
    #audio_n = audio * 10000
    #audio_r = np.round(audio_n, 2)
    show = True
    if show:
        print(audio_norm)
        print(len(audio_norm))
        plt.plot(audio_norm[:200], '.', ms=0.3, c='r')
        plt.title('Original Array')
        plt.show()
    return audio_norm[:200]

#input path
path="data/messesteller1.wav"


#plot_audio(path)
#spectrogram(path)

#sample two audio files
#file1 = get_audio_data(path, 100)






