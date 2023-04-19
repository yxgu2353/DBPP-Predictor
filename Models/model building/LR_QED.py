# _*_ coding: utf-8 _*_
# @Time : $[DATE] $[TIME]
# @Author : G

from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, \
    confusion_matrix, balanced_accuracy_score, f1_score
from itertools import product
import pickle


parameters_LR = {'penalty': ['l1', 'l2', 'none'], 'C': np.logspace(-5, 9, 8, base=2)}
model_map = {'LR': LR}
parameter_grid = {'LR': parameters_LR}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)

# evaluation metrics
def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, pos_label=1, average="binary")


def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1, average="binary")


def auc(y_true, y_scores):
    return roc_auc_score(y_true, y_scores)


def mcc(y_true, y_pred):
    return matthews_corrcoef(y_true, y_pred)


def new_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def sp(y_true, y_pred):
    cm = new_confusion_matrix(y_true, y_pred)
    return cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])

def balanced_acc(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)

def f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred)

# Find the best parameter using Training data
def classify_generator(tuned_parameters, method, train_x, train_y, n_jobs=-1):
    """
    Return the best model and the parameters of the model'
    """
    grid = GridSearchCV(method(), param_grid=tuned_parameters, scoring="accuracy", cv=cv, n_jobs=n_jobs)
    grid.fit(train_x, train_y)
    return grid.best_estimator_, grid.best_params_

def data_reader(filename):
    """
    Read QED file
    """
    data = pd.read_csv(filename, header=None).values
    x = data[:, 1:]
    y = data[:, 0]
    return x, y

# Train & test
# The folds are made by preserving the percentage of samples for each class
def train_results(best_model, train_x, train_y, ):
    """ Return the performance of the cross-validation -- Part of train """
    # TRAIN
    y_true_train = []
    y_pred_train = []
    y_score_train = []
    ACC_TRAIN = []
    PRECISION_TRAIN = []
    RECALL_TRAIN = []
    AUC_TRAIN = []
    MCC_TRAIN = []
    SP_TRAIN = []
    for train_index, test_index in cv.split(train_x, train_y):
        x_train, x_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        m = best_model.fit(x_train, y_train)
        # The train-result of every fold
        y_true_train.extend(y_train)
        y_pred_train.extend(m.predict(x_train))
        y_score_train.extend(m.predict_proba(x_train)[:, 1])

        # Gather the KFold performance for every fold
        ACC_train = accuracy(y_true_train, y_pred_train)
        Precision_train = precision(y_true_train, y_pred_train)
        Recall_train = recall(y_true_train, y_pred_train)
        AUC_train = auc(y_true_train, y_score_train)
        MCC_train = mcc(y_true_train, y_pred_train)
        SP_train = sp(y_true_train, y_pred_train)
        ACC_TRAIN.append(ACC_train)
        PRECISION_TRAIN.append(Precision_train)
        RECALL_TRAIN.append(Recall_train)
        AUC_TRAIN.append(AUC_train)
        MCC_TRAIN.append(MCC_train)
        SP_TRAIN.append(SP_train)
        # Show this performance
        print('Train_results ACC: {:.4f} Precision: {:.4f} Recall: {:.4f} AUC: {:.4f} MCC: {:.4f} SP: {:.4f}'.format(ACC_train,
                                                                                                                     Precision_train,
                                                                                                                     Recall_train,
                                                                                                                     AUC_train,
                                                                                                                     MCC_train,
                                                                                                                     SP_train))
    print('-'*30 + 'Train_results' + '-'*30)
    # Save train result
    Train_performance = pd.DataFrame({'ACC_TRAIN': ACC_TRAIN,
                                      'PRECISION_TRAIN': PRECISION_TRAIN,
                                      'RECALL_TRAIN': RECALL_TRAIN,
                                      'AUC_TRAIN': AUC_TRAIN,
                                      'MCC_TRAIN': MCC_TRAIN,
                                      'SP_TRAIN': SP_TRAIN})
    # Train_performance.to_csv('Train_result.csv', mode='a', index=False)

    return accuracy(y_true_train, y_pred_train), precision(y_true_train, y_pred_train), recall(y_true_train, y_pred_train), auc(y_true_train, y_score_train),\
           mcc(y_true_train, y_pred_train), sp(y_true_train, y_pred_train), y_true_train, y_pred_train

def cv_results(best_model, train_x, train_y):
    """ Return the performance of the cross-validation -- Part of test """
    # TEST
    y_true_test = []
    y_pred_test = []
    y_score_test = []
    ACC_TEST = []
    PRECISION_TEST = []
    RECALL_TEST = []
    AUC_TEST = []
    MCC_TEST = []
    SP_TEST = []
    for train_index, test_index in cv.split(train_x, train_y):
        x_train, x_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        m = best_model.fit(x_train, y_train)


        # The test-result of every fold
        y_true_test.extend(y_test)
        y_pred_test.extend(m.predict(x_test))
        y_score_test.extend(m.predict_proba(x_test)[:, 1])

        # Gather the KFold performance for every fold
        ACC_test = accuracy(y_true_test, y_pred_test)
        Precision_test = precision(y_true_test, y_pred_test)
        Recall_test = recall(y_true_test, y_pred_test)
        AUC_test = auc(y_true_test, y_score_test)
        MCC_test = mcc(y_true_test, y_pred_test)
        SP_test = sp(y_true_test, y_pred_test)
        ACC_TEST.append(ACC_test)
        PRECISION_TEST.append(Precision_test)
        RECALL_TEST.append(Recall_test)
        AUC_TEST.append(AUC_test)
        MCC_TEST.append(MCC_test)
        SP_TEST.append(SP_test)
        # Show this performance
        print('Valid_results ACC: {:.4f} Precision: {:.4f} Recall: {:.4f} AUC: {:.4f} MCC: {:.4f} SP: {:.4f}'.format(
            ACC_test,
            Precision_test,
            Recall_test,
            AUC_test,
            MCC_test,
            SP_test))
    print('*' * 30 + 'Test_results' + '*' * 30)


    # Save test result
    Test_performance = pd.DataFrame({'ACC_TEST': ACC_TEST,
                                      'PRECISION_TEST': PRECISION_TEST,
                                      'RECALL_TEST': RECALL_TEST,
                                      'AUC_TEST': AUC_TEST,
                                      'MCC_TEST': MCC_TEST,
                                      'SP_TEST': SP_TEST})
    # Test_performance.to_csv('Valid_result.csv', mode='a', index=False)
    return accuracy(y_true_test, y_pred_test), precision(y_true_test, y_pred_test), recall(y_true_test, y_pred_test), auc(y_true_test, y_score_test),\
           mcc(y_true_test, y_pred_test), sp(y_true_test, y_pred_test), y_true_test, y_pred_test

def test_results(best_model, test_x, test_y):
    """Return the performance of the test validation"""
    y_true = test_y
    y_pred = best_model.predict(test_x)
    y_scores = best_model.predict_proba(test_x)[:, 1]
    return accuracy(y_true, y_pred), precision(y_true, y_pred), recall(y_true, y_pred), auc(y_true, y_scores), mcc(
        y_true, y_pred), sp(y_true, y_pred), y_true, y_pred

def search_best_model(training_data, method_name, test_data):
    'Return {"model": best_model, "cv": metrics_of_cv,"tv": metrics_of_test,"parameter": best_parameter", "method": method_name}'
    train_x = training_data[0]
    train_y = training_data[1]
    test_x = test_data[0]
    test_y = test_data[1]
    tuned_parameters = parameter_grid[method_name]
    method = model_map[method_name]
    cg = classify_generator(tuned_parameters, method, train_x, train_y)
    best_model = cg[0]
    best_parameter = cg[1]
    train_metrics = train_results(best_model, train_x, train_y)[:6]
    cv_metrics = cv_results(best_model, train_x, train_y)[:6]
    TV_metrics = test_results(best_model, test_x, test_y)[:6]
    result = {'model': best_model, 'cv': cv_metrics, 'tv': TV_metrics, 'train': train_metrics, 'method': method_name}
    return result, best_model, best_parameter

def train_main(model_list, fp_list, train_name, test_name):
    model_names = product(model_list, fp_list, train_name, test_name)
    train_metrics_file = open('QED_train_metrics.txt', 'w')
    cv_metrics_file = open('QED_cv_metrics.txt', 'w')
    tv_metrics_file = open('QED_tv_metrics.txt', 'w')
    parameter_file = open("QED_parameter.txt", "w")
    train_metrics_file.write('Target\tFingerprint\tMethod\tAccuracy\tPrecision\tRecall\tAUC\tMCC\tSP\n')
    cv_metrics_file.write('Target\tFingerprint\tMethod\tAccuracy\tPrecision\tRecall\tAUC\tMCC\tSP\n')
    tv_metrics_file.write('Target\tFingerprint\tMethod\tAccuracy\tPrecision\tRecall\tAUC\tMCC\tSP\n')
    parameter_file.write("Method\tParameter\n")
    for method_name, fp_name, train_target, test_target in model_names:
        print(method_name, fp_name, train_target, test_target)
        training_data = data_reader(train_target+'_'+fp_name+'.csv')
        test_data = data_reader(test_target+'_'+fp_name+'.csv')
        model_results = search_best_model(training_data, method_name, test_data)[0]
        saved_model = search_best_model(training_data, method_name, test_data)[1]
        pickle.dump(saved_model, open('QED_' + method_name + '_' + fp_name + '.model', 'wb'))
        print('>>>>>>>--- Saving the model -ing --->>>>>>>')
        train_res = [str(x) for x in model_results['train']]
        cv_res = [str(x) for x in model_results['cv']]
        tv_res = [str(x) for x in model_results['tv']]
        train_metrics_file.write('%s\t%s\t%s\t%s\n' % (train_target, fp_name, method_name, '\t'.join(train_res)))
        cv_metrics_file.write('%s\t%s\t%s\t%s\n' % (train_target, fp_name, method_name, '\t'.join(cv_res)))
        tv_metrics_file.write('%s\t%s\t%s\t%s\n' % (test_target, fp_name, method_name, '\t'.join(tv_res)))
        parameter_file.write('%s\t%s\n' % (method_name, search_best_model(training_data, method_name, test_data)[2]))
    train_metrics_file.close()
    cv_metrics_file.close()
    tv_metrics_file.close()

if __name__ == '__main__':
    model_list = ['LR']
    fp_list = ['QED', 'QED_Norm']
    train_target = ['Training_Sample01']
    test_target = ['Valid_Sample01']
    train_main(model_list, fp_list, train_target, test_target)
    print('Have Finished!')