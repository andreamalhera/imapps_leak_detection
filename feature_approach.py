import os
import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pylab
import PIL
from PIL import Image
from matplotlib import pyplot as plt
#import cv2

ouput_path = "data/feature_approach_data"
no_leak_files_path = "data/train_wavs"
leak_files_path = "data/leak_data_for_testing"
leak_output_path = "data/feature_approach_data/leak"
no_leak_output_path = "data/feature_approach_data/no_leak"


def save_histograms(files_path, output_path):
    for file in os.scandir(files_path):
        plt.figure(figsize=(10, 4))
        print("File:", str(file))
        signal, samplerate = librosa.core.load(file)
        mel_matrix = librosa.feature.melspectrogram(y=signal, sr=samplerate, n_mels=64, fmax=8000)
        librosa.display.specshow(librosa.power_to_db(mel_matrix, ref=np.max))
        plt.tight_layout()
        plt.savefig(output_path + "/" + str(file.name)[:-4] + ".png")

def histogram(image_path, filename):
    img = cv2.imread(image_path, -1)
    color = ('b', 'g', 'r')
    for channel, col in enumerate(color):
        histr = cv2.calcHist([img], [channel], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.title('Histogram for color scale picture')
    plt.savefig(ouput_path + "/" + filename)
    return

def look_for_color(image_path):
    im = Image.open(image_path)
    im.getcolors(256)
    print(im.format)
    #print(str(colors))


if __name__ == '__main__':
    #save_histograms(leak_files_path, leak_output_path)
    #save_histograms(no_leak_files_path, no_leak_output_path)
    leak_file = "2019-06-06_00-04-07-603.png"
    no_leak_file = "2019-07-17_00-32-57-236.png"
    histogram(leak_output_path + '/' + leak_file, leak_file)
    histogram(no_leak_output_path + '/' + no_leak_file, no_leak_file)
    look_for_color(leak_output_path + '/' + leak_file)
    look_for_color(no_leak_output_path + '/' + no_leak_file)


