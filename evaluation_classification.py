def compute_roc_auc(model, classification_results, labels):
    pass

def compute_f1(model, classification_results, labels):
    pass

def compute_mmr(model, classification_results, labels):
    pass

def compare_models(models):
    pass

def classify(MODEL_NAME):


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