# _*_ coding: utf-8 _*_
# @Time : $[DATE] $[TIME]
# @Author : G

import pandas as pd
import numpy as np
import os
import random
import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dgllife.model import model_zoo
from dgllife.utils import smiles_to_bigraph
from QSAR_eval import Meter
from dgllife.utils import CanonicalAtomFeaturizer
from dgllife.utils import CanonicalBondFeaturizer
from dgllife.data import MoleculeCSVDataset
from functools import partial
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, balanced_accuracy_score, confusion_matrix
from torch.optim import Adam
from dgllife.utils import EarlyStopping
from argparse import ArgumentParser

# Device Choose
if torch.cuda.is_available():
    torch.device('cuda:0')
    device = 'cuda'
    print('='*30 + 'use GPU' + '='*30)

else:
    torch.device('cpu')
    device = 'cpu'
    print('='*30 + 'use CPU' + '='*30)

# Setting random seed
seed = 5
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

torch.manual_seed(seed)            # for CPU setting random seed
torch.cuda.manual_seed(seed)       # for GPU setting random seed
torch.cuda.manual_seed_all(seed)

# Batching a list of datapoint for dataloader
def collate_graphs(data):
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return smiles, bg, labels, masks

# Feature initialize
node_feature = CanonicalAtomFeaturizer(atom_data_field='hv')
bond_feature = CanonicalBondFeaturizer(bond_data_field='he')
n_feats = node_feature.feat_size('hv')
e_feats = bond_feature.feat_size('he')
print('-'*10 + 'Node feature count: ' + '-'*10, n_feats)
print('-'*10 + 'Edge feature count: ' + '-'*10, e_feats)

# Loading dataset
def load_data(data) -> 'DGLGraph Form':
    task = data.columns.values.tolist()
    task.remove('SMILES')
    dataset = MoleculeCSVDataset(df=data,
                                 smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                                 node_featurizer=node_feature,
                                 edge_featurizer=None,
                                 smiles_column='SMILES',
                                 cache_file_path='result/GAT/DGLGraph.bin',
                                 n_jobs=-1,
                                 init_mask=True)
    return dataset


# Train Setting
def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    Losses_train = []
    Epoches = []
    Steps = []
    y_pred_train = []
    y_true_train = []
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        if len(smiles) == 1:
            # Avoid potential issues with batch normalization
            continue

        labels, masks = labels.to(device), masks.to(device)
        n_feats = bg.ndata.pop('hv').to(device)
        logits = model(bg, n_feats)
        # Mask non-existing labels
        loss = (loss_criterion(logits, labels) * (masks != 0).float()).mean()
        pred = torch.sigmoid(logits)
        y_pred_train.extend(pred.detach().cpu().numpy())
        y_true_train.extend(labels.detach().cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(logits, labels, masks)

        if batch_id % args['print_every'] == 0:
            print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
                epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
        Losses_train.append(loss.item())
        Epoches.append(epoch)
        Steps.append(batch_id)
        score_train = []
        pred_train = []
        true_train = []
        for item in y_pred_train:
            for i in item:
                score_train.append(i)
                if i >= 0.5:
                    pred_train.append(1)
                else:
                    pred_train.append(0)

        for item in y_true_train:
            for i in item:
                true_train.append(i)
        acc_train = accuracy_score(true_train, pred_train)
        precision_train = precision_score(true_train, pred_train, pos_label=1, average="binary")
        recall_train = recall_score(true_train, pred_train, pos_label=1, average="binary")
        auc_train = roc_auc_score(true_train, score_train)
        f1_train = f1_score(true_train, pred_train)
        BACC_train = balanced_accuracy_score(true_train, pred_train)
        cm_train = confusion_matrix(true_train, pred_train, labels=[0, 1])
        SP_train = cm_train[0, 0] * 1.0 / (cm_train[0, 0] + cm_train[0, 1])
        print("Train_Results ACC: {:.4f} AUC: {:.4f} precision: {:.4f} recall: {:.4f} f1 score: {:.4f} Bacc: {:.4f} SP: {:.4f}".format(acc_train, auc_train, precision_train, recall_train, f1_train, BACC_train, SP_train))

    # train results
    Train_performance = pd.DataFrame({'Epoch': Epoches,
                                      'Steps': Steps,
                                      'loss_score': Losses_train,
                                      'train_acc': acc_train,
                                      'train_auc': auc_train,
                                      'train_precision': precision_train,
                                      'train_recall': recall_train,
                                      'train_f1': f1_train,
                                      'train_Bacc': BACC_train,
                                      'train_SP': SP_train
                                      })
    Train_performance.to_csv('result/GAT/GAT_Train_result.csv', mode='a', index=False)

    train_score = np.mean(train_meter.compute_metric('roc_auc_score'))
    train_loss = np.mean(Losses_train)
    print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, args['num_epochs'], 'auc', train_score))
    return train_score, train_loss

# Valid Setting
def run_an_valid_epoch(model, data_loader, loss_criterion):
    model.eval()
    valid_meter = Meter()
    Losses_valid = []
    y_pred_valid = []
    y_true_valid = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels, masks = labels.to(device), masks.to(device)
            n_feats = bg.ndata.pop('hv').to(device)
            logits = model(bg, n_feats)
            val_loss = (loss_criterion(logits, labels) * (masks != 0).float()).mean()
            val_loss = val_loss.detach().cpu().numpy()
            Losses_valid.append(val_loss)
            pred = torch.sigmoid(logits)
            y_pred_valid.extend(pred.detach().cpu().numpy())
            y_true_valid.extend(labels.detach().cpu().numpy())
            valid_meter.update(logits, labels, masks)
            score_valid = []
            pred_valid = []
            true_valid = []
            for item in y_pred_valid:
                for i in item:
                    score_valid.append(i)
                    if i >= 0.5:
                        pred_valid.append(1)
                    else:
                        pred_valid.append(0)

            for item in y_true_valid:
                for i in item:
                    true_valid.append(i)
            acc_valid = accuracy_score(true_valid, pred_valid)
            precision_valid = precision_score(true_valid, pred_valid, pos_label=1, average="binary")
            recall_valid = recall_score(true_valid, pred_valid, pos_label=1, average="binary")
            auc_valid = roc_auc_score(true_valid, score_valid)
            f1_valid = f1_score(true_valid, pred_valid)
            BACC_valid = balanced_accuracy_score(true_valid, pred_valid)
            cm_valid = confusion_matrix(true_valid, pred_valid, labels=[0, 1])
            SP_valid = cm_valid[0, 0] * 1.0 / (cm_valid[0, 0] + cm_valid[0, 1])
            print("Valid_Results ACC: {:.4f} AUC: {:.4f} precision: {:.4f} recall: {:.4f} f1 score: {:.4f} Bacc: {:.4f} SP: {:.4f}".format(acc_valid, auc_valid, precision_valid, recall_valid, f1_valid, BACC_valid, SP_valid))

    # valid results
    dict1 = {'valid_acc': acc_valid,
             'valid_auc': auc_valid,
             'valid_precision': precision_valid,
             'valid_recall': recall_valid,
             'valid_f1': f1_valid,
             'valid_Bacc': BACC_valid,
             'valid_SP': SP_valid}

    Valid_performance = pd.DataFrame(dict1, index=[0])

    Valid_performance.to_csv('result/GAT/GAT_Valid_result.csv', mode='a', index=False)
    valid_score = np.mean(valid_meter.compute_metric('roc_auc_score'))
    valid_loss = np.mean(Losses_valid)

    return valid_score, valid_loss

# Test Setting
def run_an_test_epoch(model, data_loader, loss_criterion):
    model.eval()
    eval_meter = Meter()
    Losses_test = []
    y_pred_test = []
    y_true_test = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels, masks = labels.to(device), masks.to(device)
            n_feats = bg.ndata.pop('hv').to(device)
            logits = model(bg, n_feats)
            test_loss = (loss_criterion(logits, labels) * (masks != 0).float()).mean()
            test_loss = test_loss.detach().cpu().numpy()
            Losses_test.append(test_loss)
            pred = torch.sigmoid(logits)
            y_pred_test.extend(pred.detach().cpu().numpy())
            y_true_test.extend(labels.detach().cpu().numpy())
            eval_meter.update(logits, labels, masks)
            score_test = []
            pred_test = []
            true_test = []
            for item in y_pred_test:
                for i in item:
                    score_test.append(i)
                    if i >= 0.5:
                        pred_test.append(1)
                    else:
                        pred_test.append(0)

            for item in y_true_test:
                for i in item:
                    true_test.append(i)
            acc_test = accuracy_score(true_test, pred_test)
            precision_test = precision_score(true_test, pred_test, pos_label=1, average="binary")
            recall_test = recall_score(true_test, pred_test, pos_label=1, average="binary")
            auc_test = roc_auc_score(true_test, score_test)
            f1_test = f1_score(true_test, pred_test)
            BACC_test = balanced_accuracy_score(true_test, pred_test)
            cm_test = confusion_matrix(true_test, pred_test, labels=[0, 1])
            SP_test = cm_test[0, 0] * 1.0 / (cm_test[0, 0] + cm_test[0, 1])
            print("Test_Results ACC: {:.4f} AUC: {:.4f} precision: {:.4f} recall: {:.4f} f1 score: {:.4f} Bacc: {:.4f} SP: {:.4f}".format(acc_test, auc_test, precision_test, recall_test, f1_test, BACC_test, SP_test))

        # test results
        dict2 = {'test_acc': acc_test,
                 'test_auc': auc_test,
                 'test_precision': precision_test,
                 'test_recall': recall_test,
                 'test_f1': f1_test,
                 'test_Bacc': BACC_test,
                 'test_SP': SP_test}

        Test_performance = pd.DataFrame(dict2, index=[0])
        Test_performance.to_csv('result/GAT/GAT_Test_result.csv', mode='a', index=False)
        test_score = np.mean(eval_meter.compute_metric('roc_auc_score'))
        test_loss = np.mean(Losses_test)

    return test_score, test_loss


def main(args, train_set, valid_set, test_set):
    train_loader = DataLoader(dataset=train_set, batch_size=10000, shuffle=True, collate_fn=collate_graphs)
    valid_loader = DataLoader(dataset=valid_set, batch_size=10000, shuffle=True, collate_fn=collate_graphs)
    test_loader = DataLoader(dataset=test_set, batch_size=10000, shuffle=True, collate_fn=collate_graphs)

    node_feature = CanonicalAtomFeaturizer(atom_data_field='hv')
    n_feats = node_feature.feat_size('hv')


    # model setting
    model = model_zoo.GATPredictor(in_feats=n_feats,
                                   hidden_feats=[128],
                                   num_heads=[8],
                                   alphas=[0.240],
                                   predictor_dropout=0.335,
                                   classifier_hidden_feats=128)
    model = model.to(device)
    loss_criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = Adam(model.parameters(), lr=0.0005, weight_decay=0.002)
    stopper = EarlyStopping(patience=80, filename='result/GAT/model_best.pth', metric='roc_auc_score')

    epochs = []
    train_scores = []
    valid_scores = []
    train_losses = []
    valid_losses = []
    for epoch in range(args['num_epochs']):
        print('=' * 20 + 'Train Performance' + '=' * 20)
        train_result = run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)
        valid_result = run_an_valid_epoch(model, valid_loader, loss_criterion)
        epochs.append(epoch)
        train_scores.append(train_result[0])
        train_losses.append(train_result[1])
        valid_scores.append(valid_result[0])
        valid_losses.append(valid_result[1])
        early_stop = stopper.step(valid_scores, model)
        if epoch % 10 == 0:
            print('epoch {:d}/{:d}, validation {} {:.4f}, validation {} {:.4f}'.format(epoch + 1, args['num_epochs'], 'auc', valid_result[0], 'loss', valid_result[1]))
        if early_stop:
            break
    # Loss summary
    a = pd.DataFrame(train_scores, columns=['train_auc'])
    b = pd.DataFrame(valid_scores, columns=['validation_auc'])
    c = pd.DataFrame(train_losses, columns=['train_loss'])
    d = pd.DataFrame(valid_losses, columns=['validation_loss'])
    e = pd.concat([a, b, c, d], axis=1)
    e.to_csv('result/GAT/Training_loss_auc.csv')

    # Save model
    fn = 'result/GAT/GAT_model.pt'
    torch.save(model.state_dict(), fn)
    model.load_state_dict(torch.load(fn, map_location=torch.device('cpu')))
    best_model = model.to(device)

    test_score = run_an_test_epoch(best_model, test_loader, loss_criterion)
    print('=' * 20 + 'Test Performance' + '=' * 20)
    print('test {} {:.4f}'.format('auc', test_score[0]))

    # test metric
    y_pred_test = []
    y_true_test = []
    for batch_id, batch_data in enumerate(test_loader):
        smiles, bg, labels, masks = batch_data
        labels, masks = labels.to(device), masks.to(device)
        n_feats = bg.ndata.pop('hv').to(device)
        # e_feats = bg.edata.pop('he').to(device)
        logits = torch.sigmoid(best_model(bg, n_feats))
        y_pred_test.extend(logits.detach().cpu().numpy())
        y_true_test.extend(labels.detach().cpu().numpy())
    score = []
    pred = []
    true = []
    for item in y_pred_test:
        for i in item:
            score.append(i)
            if i >= 0.5:
                pred.append(1)
            else:
                pred.append(0)

    for item in y_true_test:
        for i in item:
            true.append(i)
    acc = accuracy_score(true, pred)
    precision = precision_score(true, pred, pos_label=1, average="binary")
    recall = recall_score(true, pred, pos_label=1, average="binary")
    auc = roc_auc_score(true, score)
    print('acc: {:.4f}, precision{:.4f}, recall{:.4f}, auc{:.4f}'.format(acc, precision, recall, auc))
    with open('result/GAT/test_result.txt', 'w') as f:
        f.write('acc:' + str(acc) + '\n' + 'precision:' + str(precision) + '\n' + 'recall:' + str(
            recall) + '\n' + 'auc:' + str(auc))
        f.write('Best val {}: {}\n'.format('auc', stopper.best_score))
        f.write('Test {}: {}\n'.format('auc', test_score[0]))

    return stopper.best_score
    # return auc

if __name__ == '__main__':
    parser = ArgumentParser('Binary Classification')
    parser.add_argument('-pe', '--print-every', type=int, default=20,
                        help='Print the training progress every X mini-batches')
    parser.add_argument('-np', '--num-epochs', type=int, default=500,
                        help='Maximum number of epochs allowed for training.')
    args = parser.parse_args().__dict__

    train_dc = pd.read_csv('Train_Sample01_1.csv')
    valid_dc = pd.read_csv('Test_Sample01_1.csv')
    test_dc = pd.read_csv('Valid_Sample01.csv')
    train_datasets = load_data(train_dc)
    valid_datasets = load_data(valid_dc)
    test_datasets = load_data(test_dc)

    # Dataset Analysis
    print('All datasets: ' + '-'*10, len(train_datasets) + len(test_datasets) + len(valid_datasets))
    print('Training sets: ' + '-'*10, len(train_datasets))
    print('Validation sets: ' + '-'*10, len(valid_datasets))
    print('test sets: ' + '-'*10, len(test_datasets))
    main(args, train_set=train_datasets, valid_set=valid_datasets, test_set=test_datasets)