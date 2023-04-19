# _*_ coding: utf-8 _*_
# @Time : $[DATE] $[TIME]
# @Author : G

import pandas as pd
import pickle
import numpy as np
import argparse
import os

# Arg Setting ======================================================
parser = argparse.ArgumentParser(description='Validation for Classification Model')
parser.add_argument('--output', type=str, default='Valid_result',
                    help='The logger output files')
parser.add_argument('--target', type=str, default='Druglike',
                    help='The target name of the models')
parser.add_argument('--valid_target', type=str, default='Valid',
                    help='The target file name to be valid')
parser.add_argument('--represent_name', type=str, default='Descriptor_Norm',
                    help='The descriptor file name to be used')
parser.add_argument('--method_name', type=str, default='GBM',
                    help='The model name to be used')
args = parser.parse_args()


def data_loader(file):
    data = pd.read_csv(file).values
    train_x = data[:, 1:]
    train_y = data[:, 0]
    return train_x, train_y

def no_label_loader(file):
    data = pd.read_csv(file).values
    train_x = data
    return train_x


def predict_validation(target, method_name, represent, valid_name):
    need_valid_data = data_loader(valid_name + '_' + represent + '.csv')
    valid_x = need_valid_data[0]
    valid_y = need_valid_data[1]
    model_file = open(target + '_' + method_name + '_' + represent + '.model', 'rb')
    model = pickle.load(model_file)
    y_pred = model.predict(valid_x)
    y_prob = model.predict_proba(valid_x)[:, 1]
    return y_pred, y_prob, valid_y

if __name__ == '__main__':
    target = args.target
    method = args.method_name
    represent = args.represent_name
    valid_target = args.valid_target
    result = predict_validation(target, method, represent, valid_target)
    Label = pd.DataFrame(np.array(result[2]), columns=['label'])
    Label_pred = pd.DataFrame(np.array(result[0]), columns=['Label_pred'])
    Prediction_Prob = pd.DataFrame(np.array(result[1]), columns=['Prob'])
    Final_result = pd.concat([Label, Label_pred, Prediction_Prob], axis=1)
    Result = Final_result.to_csv('Worlddrug_prediction_reuslt.csv', index=None)
    print('-' * 40 + 'Done!' + '-' * 40)