# IMAPPS
Innovative Mobile Applications course LMU 2019 Group A<br>
Applied deep learning on audio files <br>
Within the context of "ErLoWa – Automatisierte Erkennung und Lokalisierung von Leckstellen in Wasserversorgungsnetzen"

## Abstract
This project is part of a university's
Anomaly Detection and Localization in big industrial facilities is one important
application field of Artificial Intelligence nowadays. As part of Munich's University program
and in cooperation with Stadtwerken München I got to participate in a one week hackathon-like
project, where based on auditory data we built and evaluated different Machine Learning models to
detect and localize leaks in water systems all over the city. If you are interested in Machine
Learning topics and would like to know how a real life field use-case looks like, please feel
welcome to join this talk.

## Presentation Slides

* [Uni-Abschlussvortrag](https://github.com/andreamalhera/imapps/blob/master/presentation_slides/imapps_pra%CC%88si.pdf): [edit](https://docs.google.com/presentation/d/1NYBnsavEj0R6eonXO1-UtW5_P0nvk1D6RPbJvOmcEzw/edit)
* [Pyladies Munich](https://github.com/andreamalhera/imapps/blob/master/presentation_slides/IMAPPS_English.pdf): [edit](https://docs.google.com/presentation/d/1VSdyFK8aimThcTbeWq-KZg1s-knkO662pX8kzm7lxYo/edit#slide=id.g5efc720b3d_1_154)
* [Studenten und junge Ingeneure München](https://github.com/andreamalhera/imapps/blob/master/presentation_slides/IMAPPS_Deutsch.pdf): [edit](https://docs.google.com/presentation/d/11ssvUo7ST1yWSHPFUUC9F17VLHH1BVPZxccq4erU0_s/edit#slide=id.g5efc720b3d_1_154)


## Dokumentation: Struktur des Codes

Für jede Autoencoder-Architektur wurde eine eigene Klasse angelegt. Folgende Autoencoder existieren:
- Simple Autoencoder aus ausschließlich Dense Layers (`simple_autoencoder.py`)
- CNN Autoencoder (`cnn_autoencoder.py`, `cnn_autoencoder_more_filter.py`)
- Variational Autoencoder (`variational_autoencoder.py`, `va_autoencoder.py`, `vl_autoencoder.py`)
- LSTM Autoencoder (`lstm_autoencoder.py`, `LSTM_autoencoder_example.py`)

Zusätzlich gibt es Klassen zum Preprocessing der Daten (`preprocessing.py`, `preprocessing_leak_test_data.py`,
`model_data_preparation.py`).

Für die Klassifikation und zur Vorbereitung der Klassifikation gibt es die Klassen 
`classification_test_data.py` und `classification_test_data_preparation.py`.

Die Evaluation findet mit Hilfe der Klassen `evaluation_classification.py` und `roc_auc.py` statt.

Einige Hilfsfunktionen sind in dem Ordner `utilities` ausgelagert.

Die restlichen Klassen enthalten ebenfalls teilweise Hilfsfunktionen.


## Technologies
Project is created with:
* Python 3.6
* Keras 2.2.4 using TensorFlow backend
* librosa 0.6.3

## Installation
Pre-requirements:
- install python 
- install libav

```bash
conda create -n imapps_venv python=3
conda activate imapps_venv

#install requirements
pip install -r requirements.txt
```

## Installation GPU
- install miniconda
- add to file .zshrc_local in your home directory (maybe you need to create it first):
```bash
# Add for Miniconda3

export PATH=$HOME/miniconda3/bin:PATH
. $HOME/miniconda3/etc/profile.d/conda.sh

export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin && clear
```
- if this is not working, try to add everything except the last line to file .zshrc

- create conda environment and activate it:
```bash
conda create -n "name_of_env"
conda activate "name_of_env"
```
- install tensorflow, tensorflow-gpu, keras, librosa and matplotlib (same like requirements.txt)
- install correct version of cudatoolkit: check version with
```bash
nvcc --version
```
and google for right version. When it's something like "Cuda compilation tools, release **9.1**, V9.1.85", you can install 9.1 like this:
(from https://anaconda.org/numba/cudatoolkit)
```bash
conda install -c numba cudatoolkit 
```

