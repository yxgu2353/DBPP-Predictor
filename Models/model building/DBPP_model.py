# _*_ coding: utf-8 _*_
# @Time : $[DATE] $[TIME]
# @Author : G

import os
import argparse
from sklearn.svm import SVC
from lightgbm import LGBMClassifier as GBM
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier as nn
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, \
    confusion_matrix, balanced_accuracy_score, f1_score
from itertools import product
import pickle

# Arg Setting ======================================================
parser = argparse.ArgumentParser(description='Machine Learning for Combined Representation QSAR Model')
parser.add_argument('--seed', type=int, default=2022,
                    help='Random seed for the QSAR model"')
parser.add_argument('--output_path', type=str, default='Combined_result',
                    help='The logger output files')
parser.add_argument('--split', type=int, default=10,
                    help='K-fold cross validation setting')
parser.add_argument('--endpoint_name', type=str, default='Combined',
                    help='The Endpoint file name to be used')

args = parser.parse_args()

# Parameter Setting ======================================================
parameters_GBM = {'max_depth': range(2, 8, 1), 'num_leaves': range(2, 20, 1), 'min_child_samples': range(2, 20, 2)}
parameters_svm = {'kernel': ['rbf'], 'gamma': np.logspace(-15, 3, 10, base=2),
                  'C': np.logspace(-5, 9, 8, base=2), 'class_weight': ['balanced']}
parameters_knn = {'n_neighbors': range(3, 10, 2), "weights": ['distance', 'uniform']}
parameters_rf = {"n_estimators": range(10, 101, 10),
                 'criterion': ['gini', 'entropy'],
                 'oob_score': ['True'],
                 'class_weight': ['balanced_subsample', 'balanced']}
parameters_nn = {'learning_rate': ['constant'],
                 'max_iter': [1000],
                 'hidden_layer_sizes': [(5,), (10,), (15,), (20,), (25,), (30,), (35,), (40,), (45,)],
                 'alpha': 10.0 ** -np.arange(1, 7),
                 'activation': ['relu'],
                 'solver': ['adam']}

model_map = {'GBM': GBM, 'svm': SVC, 'knn': KNN, 'rf': rf, 'MLP': nn}
parameter_grid = {'GBM': parameters_GBM, 'svm': parameters_svm, 'knn': parameters_knn, 'MLP': parameters_nn, 'rf': parameters_rf}


# Evaluation metrics ======================================================
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

def f1(y_true, y_pred):
    return f1_score(y_true, y_pred)

# Cross Validation Set, default: 10-fold, random_state = 2022
cv = StratifiedKFold(n_splits=args.split, shuffle=True, random_state=args.seed)

# Data Load Setting ======================================================
def data_reader(file1, file2):
    """ Read data form. It sets for concat two representation types """
    Descriptor_data = pd.read_csv(file1, header=None).values
    Property_data = pd.read_csv(file2).values
    Descriptor_x = Descriptor_data[:, 1:]
    Descriptor_y = Descriptor_data[:, 0]
    Property_x = Property_data[:, 1:]
    Property_y = Property_data[:, 0]
    return Descriptor_x, Descriptor_y, Property_x, Property_y

# Model training Setting ======================================================
# Train & test &  valid
# The folds are made by preserving the percentage of samples for each class
# Model fit by (train_x, train_y), then was used for testing other data (test_set and valid_set)
def train_results(best_model, train_x, train_y):
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
    BACC_TRAIN = []
    F1_TRAIN = []
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
        BACC_train = balanced_acc(y_true_train, y_pred_train)
        f1_train = f1(y_true_train, y_pred_train)
        ACC_TRAIN.append(ACC_train)
        PRECISION_TRAIN.append(Precision_train)
        RECALL_TRAIN.append(Recall_train)
        AUC_TRAIN.append(AUC_train)
        MCC_TRAIN.append(MCC_train)
        SP_TRAIN.append(SP_train)
        BACC_TRAIN.append(BACC_train)
        F1_TRAIN.append(f1_train)
        # Show this performance
        print('Train_results ACC: {:.4f} Precision: {:.4f} Recall: {:.4f} AUC: {:.4f} MCC: {:.4f} SP: {:.4f} BACC: {:.4f} f1: {:.4f}'.format(ACC_train,
                                                                                                                                             Precision_train,
                                                                                                                                             Recall_train,
                                                                                                                                             AUC_train,
                                                                                                                                             MCC_train,
                                                                                                                                             SP_train,
                                                                                                                                             BACC_train,
                                                                                                                                             f1_train))
    print('-'*30 + 'Train_results' + '-'*30)
    # Save train result
    Train_performance = pd.DataFrame({'ACC_TRAIN': ACC_TRAIN,
                                      'PRECISION_TRAIN': PRECISION_TRAIN,
                                      'RECALL_TRAIN': RECALL_TRAIN,
                                      'AUC_TRAIN': AUC_TRAIN,
                                      'MCC_TRAIN': MCC_TRAIN,
                                      'SP_TRAIN': SP_TRAIN,
                                      'BACC_TRAIN': BACC_TRAIN,
                                      'F1_TRAIN': F1_TRAIN})

    train_performance_file = os.path.join(args.output_path, 'Train_performance_file.csv')
    Train_performance.to_csv(train_performance_file, mode='a', index=False)

    return accuracy(y_true_train, y_pred_train), precision(y_true_train, y_pred_train), recall(y_true_train, y_pred_train), auc(y_true_train, y_score_train),\
           mcc(y_true_train, y_pred_train), sp(y_true_train, y_pred_train), balanced_acc(y_true_train, y_pred_train), f1(y_true_train, y_pred_train)


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
    BACC_TEST = []
    F1_TEST = []
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
        BACC_test = balanced_acc(y_true_test, y_pred_test)
        f1_test = f1(y_true_test, y_pred_test)

        ACC_TEST.append(ACC_test)
        PRECISION_TEST.append(Precision_test)
        RECALL_TEST.append(Recall_test)
        AUC_TEST.append(AUC_test)
        MCC_TEST.append(MCC_test)
        SP_TEST.append(SP_test)
        BACC_TEST.append(BACC_test)
        F1_TEST.append(f1_test)
        # Show this performance
        print('Valid_results ACC: {:.4f} Precision: {:.4f} Recall: {:.4f} AUC: {:.4f} MCC: {:.4f} SP: {:.4f} BACC: {:.4f} F1: {:.4f}'.format(
            ACC_test,
            Precision_test,
            Recall_test,
            AUC_test,
            MCC_test,
            SP_test,
            BACC_test,
            f1_test))
    print('*' * 30 + 'Test_results' + '*' * 30)


    # Save test result
    Test_performance = pd.DataFrame({'ACC_TEST': ACC_TEST,
                                      'PRECISION_TEST': PRECISION_TEST,
                                      'RECALL_TEST': RECALL_TEST,
                                      'AUC_TEST': AUC_TEST,
                                      'MCC_TEST': MCC_TEST,
                                      'SP_TEST': SP_TEST,
                                      'BACC_TEST': BACC_TEST,
                                      'F1_TEST': F1_TEST})

    test_performance_file = os.path.join(args.output_path, 'Test_performance_file.csv')
    Test_performance.to_csv(test_performance_file, mode='a', index=False)
    return accuracy(y_true_test, y_pred_test), precision(y_true_test, y_pred_test), recall(y_true_test, y_pred_test), auc(y_true_test, y_score_test),\
           mcc(y_true_test, y_pred_test), sp(y_true_test, y_pred_test), balanced_acc(y_true_test, y_pred_test), f1(y_true_test, y_pred_test)


def valid_results(best_model, test_x, test_y):
    """Return the performance of the test validation"""
    y_true = test_y
    y_pred = best_model.predict(test_x)
    y_scores = best_model.predict_proba(test_x)[:, 1]
    return accuracy(y_true, y_pred), precision(y_true, y_pred), recall(y_true, y_pred), auc(y_true, y_scores), mcc(
        y_true, y_pred), sp(y_true, y_pred), balanced_acc(y_true, y_pred), f1(y_true, y_pred)

# Find the best parameter using Training data
def classify_generator(tuned_parameters, method, train_x, train_y, n_jobs=-1):
    """
    Return the best model and the parameters of the model'
    """
    if method == SVC:
        grid = GridSearchCV(method(probability=True, random_state=2), param_grid=tuned_parameters, scoring="accuracy",
                            cv=cv, n_jobs=n_jobs)
    elif method == KNN:
        grid = GridSearchCV(method(), param_grid=tuned_parameters, scoring="accuracy", cv=cv, n_jobs=n_jobs)
    else:
        grid = GridSearchCV(method(random_state=5), param_grid=tuned_parameters, scoring="accuracy", cv=cv,
                            n_jobs=n_jobs)
    grid.fit(train_x, train_y)
    return grid.best_estimator_, grid.best_params_


def search_best_model(method_name, training_data, valid_data):
    'Return {"model": best_model, "cv": metrics_of_cv,"tv": metrics_of_test,"parameter": best_parameter", "method": method_name}'

    train_x = np.concatenate((training_data[0], training_data[2]), axis=1)
    train_y = training_data[1]
    test_x = np.concatenate((valid_data[0], valid_data[2]), axis=1)
    test_y = valid_data[1]
    tuned_parameters = parameter_grid[method_name]
    method = model_map[method_name]
    cg = classify_generator(tuned_parameters, method, train_x, train_y)
    best_model = cg[0]
    best_parameter = cg[1]
    train_metrics = train_results(best_model, train_x, train_y)[:8]
    cv_metrics = cv_results(best_model, train_x, train_y)[:8]
    TV_metrics = valid_results(best_model, test_x, test_y)[:8]
    result = {'model': best_model, 'cv': cv_metrics, 'tv': TV_metrics, 'train': train_metrics, 'method': method_name}
    return result, best_model, best_parameter


def train_main(model, descriptor, properties, training_name, valid_name, save_file=args.output_path):
    result_dir = os.path.join(save_file)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    model_names = product(model, descriptor, properties, training_name, valid_name)

    train_result = os.path.join(save_file, f'{args.endpoint_name}_train_metrics.txt')
    cv_result = os.path.join(save_file, f'{args.endpoint_name}_cv_metrics.txt')
    valid_result = os.path.join(save_file, f'{args.endpoint_name}_valid_metrics.txt')
    parameter_result = os.path.join(save_file, f'{args.endpoint_name}_parameter.txt')

    train_metrics_file = open(train_result, 'w')
    cv_metrics_file = open(cv_result, 'w')
    tv_metrics_file = open(valid_result, 'w')
    parameter_file = open(parameter_result, "w")
    train_metrics_file.write('Target\tDescriptor\tProperty\tMethod\tAccuracy\tPrecision\tRecall\tAUC\tMCC\tSP\tBACC\tF1\n')
    cv_metrics_file.write('Target\tDescriptor\tProperty\tMethod\tAccuracy\tPrecision\tRecall\tAUC\tMCC\tSP\tBACC\tF1\n')
    tv_metrics_file.write('Target\tDescriptor\tProperty\tMethod\tAccuracy\tPrecision\tRecall\tAUC\tMCC\tSP\tBACC\tF1\n')
    parameter_file.write("Method\tParameter\n")
    for method_name, descriptor_name, property_name, train_target, valid_target in model_names:
        training_data = data_reader(train_target + '_' + descriptor_name + '.csv',
                                    train_target + '_' + property_name + '.csv')
        valid_data = data_reader(valid_target + '_' + descriptor_name + '.csv',
                                 valid_target + '_' + property_name + '.csv')

        model_results = search_best_model(method_name, training_data, valid_data)[0]
        saved_model = search_best_model(method_name, training_data, valid_data)[1]
        model_result = os.path.join(save_file, f'{args.endpoint_name}_' + method_name + '_' + descriptor_name + '_' + property_name + '.model')
        pickle.dump(saved_model, open(model_result, 'wb'))
        print('>>>>>>>--- Saving the model -ing --->>>>>>>')
        train_res = [str(x) for x in model_results['train']]
        cv_res = [str(x) for x in model_results['cv']]
        tv_res = [str(x) for x in model_results['tv']]
        train_metrics_file.write('%s\t%s\t%s\t%s\t%s\n' % (train_target, descriptor_name, property_name, method_name, '\t'.join(train_res)))
        cv_metrics_file.write('%s\t%s\t%s\t%s\t%s\n' % (train_target, descriptor_name, property_name, method_name, '\t'.join(cv_res)))
        tv_metrics_file.write('%s\t%s\t%s\t%s\t%s\n' % (valid_target, descriptor_name, property_name, method_name, '\t'.join(tv_res)))
        parameter_file.write('%s\t%s\n' % (method_name, search_best_model(method_name, training_data, valid_data)[2]))
    train_metrics_file.close()
    cv_metrics_file.close()
    tv_metrics_file.close()

if __name__ == '__main__':
    # model_list = ['svm', 'knn', 'rf', 'GBM']
    model_list = ['svm', 'GBM']
    descriptor_list = ['Descriptor_Norm', 'PC']
    property_list = ['MACCS_Property_Pred', 'MACCS_Property_Prob', 'Morgan_Property_Pred', 'Morgan_Property_Prob']
    train_name = ['Training_Sample01']
    valid_name = ['Valid_Sample01']
    train_main(model_list, descriptor_list, property_list, train_name, valid_name)
    print('Have Finished!')



