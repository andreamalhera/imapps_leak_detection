from sklearn.metrics import f1_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import itertools
from classification_test_data import classify

import pdb

def compute_roc_auc(classification_results, labels, save_path):
    # fpr(false positive rate), tpr(true positive rate)
    fpr, tpr, _ = roc_curve(labels, classification_results)
    auc = roc_auc_score(labels, classification_results)
    roc_auc = dict()
    roc_auc['roc'] = (fpr, tpr)
    roc_auc['auc'] = auc

    # plot roc curve
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr, tpr, marker='.')
    plt.savefig(save_path+'-roc.png', dpi=800)
    return roc_auc

def compute_mmr(classification_results, labels):
    pass


def plot_confusion_matrix(cm,
                          target_names,
                          save_path,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(save_path+'-confusion_mat.png', dpi=800)


def compare_models(models):
    pass

def run_classification_evaluation():
    '''
    To evaluate the quality of the classification we compare the classification result (leak/ no_leak)
    for a test data set with their true labels using computed metrics (roc_auc, f1 score and mean reciprocal rank)

    We compare the computed metrics between models

    We plot computed metrics for different models
    We save the plot

    Classifier output:

    classifier_output = {file_name: (true_label,prediction_label), file_name: (true_label,prediction_label),
    file_name: (true_label,prediction_label)}

    E.g.
    classifier_output = {file1: (0,1), file2: (1,1), file3: (0,0)}
    where 1 is leak and 0 is no_leak.

    '''
    models=[]
    model_results = dict()

    weight_file_name = "SIMPLE_weights_e30_dim10_ba64_2019-08-01-10:28:10.h5"
    #weight_file_name = "CNN_more_filter_weights_e100_dim2_ba64.h5"
    classifier_output = classify(weight_file_name)

    labels = [i[2] for i in classifier_output]
    classification_results = [i[1] for i in classifier_output]

    roc_auc = compute_roc_auc(classification_results, labels, weight_file_name)
    print(roc_auc['auc'])
    
    # Round scores for f1, confusion
    classification_results = [round(elem) for elem in classification_results]

    model_f1_score = f1_score(labels, classification_results)
    print(model_f1_score)

    # Compute confusion matrix
    cm = confusion_matrix(labels, classification_results)
    plot_confusion_matrix(cm, target_names=['no_leak', 'leak'], save_path=weight_file_name)


if __name__ == '__main__':
    run_classification_evaluation()
    exit()