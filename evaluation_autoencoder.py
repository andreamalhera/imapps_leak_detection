from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

from simplest_autoencoder import run_simple_autoencoder


def build_keras_model():
    model = Sequential()
    model.add(Dense(20, input_dim=20, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def plot_roc_single_curve():
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

def plot_roc():
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    # Zoom in view of the upper left corner.
    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (zoomed in at top left)')
    plt.legend(loc='best')
    plt.show()


def run_example_models(X_train, y_train):
    # Keras model
    keras_model = build_keras_model()
    keras_model.fit(X_train, y_train, epochs=5, batch_size=100, verbose=1)

    # Supervised transformation based on random forests
    rf = RandomForestClassifier(max_depth=3, n_estimators=10)
    rf.fit(X_train, y_train)

    return keras_model, rf


def compute_auc(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    aucurve = auc(fpr, tpr)
    return fpr, tpr, aucurve

def run_example_models(X_train, y_train):
    # Keras model
    keras_model = build_keras_model()
    keras_model.fit(X_train, y_train, epochs=5, batch_size=100, verbose=1)

    # Supervised transformation based on random forests
    rf = RandomForestClassifier(max_depth=3, n_estimators=10)
    rf.fit(X_train, y_train)

    return keras_model, rf

def compute_auc(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    aucurve = auc(fpr, tpr)
    return fpr, tpr, aucurve

def run_example_models(X_train, y_train):
    # Keras model
    keras_model = build_keras_model()
    keras_model.fit(X_train, y_train, epochs=5, batch_size=100, verbose=1)

    # Supervised transformation based on random forests
    rf = RandomForestClassifier(max_depth=3, n_estimators=10)
    rf.fit(X_train, y_train)

    return keras_model, rf


def compute_auc(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    aucurve = auc(fpr, tpr)
    return fpr, tpr, aucurve

# TODO integrate encoder, decoder logic
if __name__ == '__main__':
    # encoder, decoder = run_simple_autoencoder()
    X, y = make_classification(n_samples=80000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train,
                                                                y_train,
                                                                test_size=0.5)
    keras_model, rf = run_example_models(X_train, y_train)

    y_pred_rf = rf.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, auc_rf = compute_auc(y_test, y_pred_rf)

    y_pred_keras = keras_model.predict(X_test).ravel()
    fpr_keras, tpr_keras, auc_keras = compute_auc(y_test, y_pred_keras)

    plot_roc_single_curve()
    # plot_roc()
    # TODO: Keras visualization broken. I think build_keras_model refactoring broken.
    # plot_roc(fpr_keras, tpr_keras, auc_keras)
