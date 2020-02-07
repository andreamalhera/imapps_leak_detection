def compute_roc_auc(model, classification_results, labels):
    pass

def compute_f1(model, classification_results, labels):
    pass

def compute_mmr(model, classification_results, labels):
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

    return [1,0,1,0]

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

    model= None
    classification_results=[0,1,0,1]
    LABELS= [1,0,0,1]

    models=[]

    compute_roc_auc(model, classification_results,  LABELS)

    compute_f1(model, classification_results, labels)
    compute_mmr(model, classification_results, labels)

    compare_models(models)




if __name__ == '__main__':
    run_classification_evaluation()
    exit()