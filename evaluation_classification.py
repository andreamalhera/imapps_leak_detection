from sklearn.metrics import f1_score, confusion_matrix, roc_curve, roc_auc_score
from matplotlib import pyplot
import numpy as np
from classification_test_data import classify

import pdb

def normalize_score(score_leak, score_no_leak):

    mean_RE_no_leak = np.mean(score_no_leak)
    std_RE_no_leak = np.std(score_no_leak)

    mean_RE_leak = np.mean(score_leak)
    std_RE_leak = np.std(score_leak)

    max_RE = max([mean_RE_leak+std_RE_leak, mean_RE_leak-std_RE_leak,
                  std_RE_no_leak+std_RE_no_leak, std_RE_no_leak-std_RE_no_leak])
    min_RE = min([mean_RE_leak+std_RE_leak, mean_RE_leak-std_RE_leak,
                  std_RE_no_leak+std_RE_no_leak, std_RE_no_leak-std_RE_no_leak])

    n_score_leak = score_leak/(max_RE-min_RE)
    n_score_no_leak = score_no_leak/(max_RE-min_RE)

    return n_score_leak, n_score_no_leak


def compute_roc_auc(classification_results, labels):
    print(classification_results)
    # fpr(false positive rate), tpr(true positive rate)
    fpr, tpr, _ = roc_curve(labels, classification_results)
    auc = roc_auc_score(labels, classification_results)
    roc_auc = dict()
    roc_auc['roc'] = (fpr, tpr)
    roc_auc['auc'] = auc

    # plot roc curve
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    pyplot.plot(fpr, tpr, marker='.')
    pyplot.savefig("CNN_more_filter_weights_e100_dim2_ba64.png", dpi=800)
    return roc_auc

def compute_mmr(classification_results, labels):
    pass

def compare_models(models):
    pass


def run_normalized_classification_evaluation():
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
    models = []
    model_results = dict()

    weight_file_name = "CNN_more_filter_weights_e100_dim2_ba64.h5"
    classifier_output = classify(weight_file_name)

    result_leak=[]
    true_leak=[]
    result_no_leak=[]
    true_no_leak=[]

    for entry in classifier_output:
        if entry[2] == 0 :
            result_no_leak.append(entry[1])
            true_no_leak.append(entry[2])
        else:
            result_leak.append(entry[1])
            true_leak.append(entry[2])

    result_leak_n, result_no_leak_n = normalize_score(result_leak, result_no_leak)
    
    classification_results= np.concatenate((result_leak_n,result_no_leak_n), axis=0)
    labels = np.concatenate((true_leak, true_no_leak),axis=None)

    roc_auc = compute_roc_auc(classification_results, labels)
    print(roc_auc['auc'])

    model_f1_score = f1_score(labels, classification_results)
    print(model_f1_score)


#    conf_mat = confusion_matrix(labels, classification_results)
#   print(confusion_matrix)


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

    weight_file_name = "CNN_more_filter_weights_e100_dim2_ba64.h5"
    classifier_output = classify(weight_file_name)
    pdb.set_trace()

    labels = [i[2] for i in classifier_output]
    classification_results = [i[1] for i in classifier_output]

    normalize_score(classification_results)

    roc_auc = compute_roc_auc(classification_results, labels)
    print(roc_auc['auc'])
    
    model_f1_score = f1_score(labels, classification_results)
    print(model_f1_score)

#    conf_mat = confusion_matrix(labels, classification_results)
 #   print(confusion_matrix)


if __name__ == '__main__':
    run_normalized_classification_evaluation()
    #run_classification_evaluation()
    exit()