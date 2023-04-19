# _*_ coding: utf-8 _*_
# @Time : $[DATE] $[TIME]
# @Author : G

import torch.nn as nn
from dgllife.model.gnn import GraphSAGE
import dgl
import torch

class GraphSAGEPredictor(nn.Module):
    def __init__(self, in_feats, hidden_feats=None, activation=None,
                 aggregator_type=None, dropout=None, classifier_hidden_feats=128, n_task=1,
                 predictor_hidden_feats=128):
        super(GraphSAGEPredictor, self).__init__()

        if predictor_hidden_feats == 128 and classifier_hidden_feats != 128:
            print('classifier_hidden_feats is deprecated and will be removed in the future, '
                  'use predictor_hidden_feats instead')
            predictor_hidden_feats = classifier_hidden_feats

        self.gnn = GraphSAGE(in_feats=in_feats,
                             hidden_feats=hidden_feats,
                             activation=activation,
                             aggregator_type=aggregator_type,
                             dropout=dropout)

        # gnn_out_feats = self.gnn.hidden_feats[-1]

        self.predict = nn.Sequential(
            # nn.Linear(hidden_feats[0] + 2, 128),
            nn.Linear(hidden_feats[0], 128),
            # nn.Linear(hidden_feats[0], 130),
            # nn.Linear(len(rdkit_feats), 128),

            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, n_task),
            # nn.ReLU(),
            # nn.Softmax(dim=1)
        )
        # self.predict = nn.Sequential(
        #     # nn.LSTMCell(graph_feat_size+rdkitEF_size, 100),
        #     nn.LeakyReLU(),
        #     nn.LayerNorm(hidden_feats[0] + 2),
        #     nn.Linear(hidden_feats[0] + 2, 64),
        #     nn.LeakyReLU(),
        #     nn.LayerNorm(64),
        #     nn.Linear(64, n_task),
        # )


    def forward(self, bg, feats):
        node_feats = self.gnn(bg, feats)
        with bg.local_scope():
            bg.ndata['hv'] = node_feats
            # Calculate graph representation by average readout.
            hg = dgl.max_nodes(bg, 'hv')
            _fgt = hg.detach().cpu().numpy()
            # Concat the graph feature and rdkit feature
            Concat_hg = torch.cat([hg], dim=1)
            Final_feature = self.predict(Concat_hg)
            # print(Final_feature.ndim)
            # Final_feature = Final_feature.reshape(-1, 1)
            # return Final_feature.squeeze(1).type(torch.LongTensor)
            # return Final_feature.squeeze(1).type(torch.LongTensor)
            return Final_feature