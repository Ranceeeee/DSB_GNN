import os.path as osp
import os
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from data_make import DSB
from torch_geometric.data import DataLoader, NeighborSampler
from torch_geometric.nn import GATConv, GNNExplainer
from torch_geometric.nn import JumpingKnowledge
import csv
import time
from tqdm import tqdm
import numpy as np
import copy
from sklearn import svm
import dgl
from gnn_explainer import GNNExplainer
import argparse
from gnn_explainer import GNNExplainer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.utils import to_undirected
from torch_geometric.utils import add_self_loops

import matplotlib.pyplot as plt

# settings

parser = argparse.ArgumentParser(description='GNN baselines')
parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
parser.add_argument('--test_chr', type=int, default=1,
                        help='which chromosome used to be test')
parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='learning_rate')
parser.add_argument('--epoch', type=int, default=100,
                        help='training_epoch')
parser.add_argument('--sleep_time', type=int, default=0,
                        help='wait for run')

args = parser.parse_args()

test_chr = args.test_chr
epoch_num = args.epoch
learning_rate = args.learning_rate
device_num = args.device

time.sleep(args.sleep_time)

starttime = time.time()
device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')

train_list = []
print('test_chr:', test_chr)

data = []
for i in range(1, 24):

    g = dgl.load_graphs("/data/NHEK_dgl_data/chr_" + str(i) + ".dgl")[0][0]
    # g = dgl.remove_self_loop(g)
    g.ndata['node_id'] = g.nodes().reshape(g.nodes().shape[0], 1)
    g_density = np.loadtxt('/data/NHEK/Node_EpiFeature_5000_2/chr' + str(i) + '.density.txt')
    g.ndata['density'] = torch.from_numpy(g_density)
    density_mean_1 = np.nanmean(g_density[:,0])
    density_mean_2 = np.nanmean(g_density[:,1])
    nan_index_1 = torch.isnan(g.ndata['density'][:,0])
    nan_index_2 = torch.isnan(g.ndata['density'][:,1])
    g.ndata['density'][:, 0][nan_index_1] = density_mean_1
    g.ndata['density'][:, 1][nan_index_2] = density_mean_2
    data.append(g)

if test_chr == 23:
    valid_chr = 1
    g_test = data[22]
    data.pop(22)
    g_valid = data[0]
    data.pop(0)

else:
    valid_chr = test_chr + 1
    g_test = data[test_chr - 1]
    g_valid = data[valid_chr - 1]
    data.pop(test_chr - 1)
    data.pop(test_chr - 1)

# preprocess
train_feats = torch.zeros(0, g.ndata['seq_feature'].shape[1])
train_density = torch.zeros(0,2)
train_edge_feats = torch.zeros(0, 1)

for graph in data:
    train_feats = torch.cat([train_feats, graph.ndata['seq_feature']], 0)
    train_density = torch.cat([train_density, graph.ndata['density']], 0)
    # train_edge_feats = torch.cat([train_edge_feats, graph.edata['edge_feature'].view(graph.edata['edge_feature'].shape[0], 1)], 0)

scaler = preprocessing.StandardScaler().fit(train_feats)
density_scaler = preprocessing.StandardScaler().fit(train_density)
# edge_scaler = preprocessing.StandardScaler().fit(train_edge_feats)

for graph in data:
    graph.ndata['feat'] = torch.from_numpy(scaler.transform(graph.ndata['seq_feature'])).float()
    graph.ndata['density_feat'] = torch.from_numpy(density_scaler.transform(graph.ndata['density'])).float()
    # graph.ndata['deg'] = graph.out_degrees().float().clamp(min=1)
    # graph.edata['edge_feature'] = graph.edata['edge_feature'].view(graph.edata['edge_feature'].shape[0], 1)
    # graph.edata['feat'] = torch.from_numpy(edge_scaler.transform(graph.edata['edge_feature'])).float()
    graph.ndata['feat'] = torch.cat((graph.ndata['feat'], graph.ndata['density_feat']), dim=1)
    N = graph.num_nodes()


g_valid.ndata['feat'] = torch.from_numpy(scaler.transform(g_valid.ndata['seq_feature'])).float()
g_valid.ndata['density_feat'] = torch.from_numpy(density_scaler.transform(g_valid.ndata['density'])).float()
g_valid.ndata['deg'] = g_valid.out_degrees().float().clamp(min=1)
# g_valid.edata['edge_feature'] = g_valid.edata['edge_feature'].view(g_valid.edata['edge_feature'].shape[0], 1)
# g_valid.edata['feat'] = torch.from_numpy(edge_scaler.transform(g_valid.edata['edge_feature'])).float()
g_valid.ndata['feat'] = torch.cat((g_valid.ndata['feat'], g_valid.ndata['density_feat']), dim=1)


g_test.ndata['feat'] = torch.from_numpy(scaler.transform(g_test.ndata['seq_feature'])).float()
g_test.ndata['density_feat'] = torch.from_numpy(density_scaler.transform(g_test.ndata['density'])).float()
g_test.ndata['deg'] = g_test.out_degrees().float().clamp(min=1)
# g_test.edata['edge_feature'] = g_test.edata['edge_feature'].view(g_test.edata['edge_feature'].shape[0], 1)
# g_test.edata['feat'] = torch.from_numpy(edge_scaler.transform(g_test.edata['edge_feature'])).float()
g_test.ndata['feat'] = torch.cat((g_test.ndata['feat'], g_test.ndata['density_feat']), dim=1)

# Net
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.line = torch.nn.Linear(1346, 32)
        self.batch_norm = torch.nn.BatchNorm1d(32)

        for i in range(1, 3):
            setattr(self, 'conv{}'.format(i), GATConv(32, 8, heads=4))
            setattr(self, 'line{}'.format(i), nn.Linear(32, 32))
            setattr(self, 'dropout{}'.format(i), nn.Dropout(p=0.5))
            setattr(self, 'batchnorm{}'.format(i), nn.BatchNorm1d(32))

        self.jk = JumpingKnowledge(mode='cat')
        self.fc = nn.Linear(32 + 2 * 32, 16)

        self.last_line = torch.nn.Linear(16, 1)


    def forward(self, x, edge_index):

        layer_out = []
        x = F.relu(self.line(x.float()))
        x = F.dropout(x, p=0.5, training=self.training)

        layer_out.append(x)

        for i in range(1, 3):
            conv = getattr(self, 'conv{}'.format(i))
            line = getattr(self, 'line{}'.format(i))
            batchnorm = getattr(self, 'batchnorm{}'.format(i))
            dropout = getattr(self, 'dropout{}'.format(i))
            x = F.elu(conv(x, edge_index) + line(x))
            # x = batchnorm(x)
            x = dropout(x)
            layer_out.append(x)

        h = self.jk(layer_out)
        h = F.relu(self.fc(h))

        h = self.last_line(h)

        h = torch.sigmoid(h).squeeze()

        return h

model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

def train(epoch, train_list, model):

    model.train()

    lenth = len(train_list)
    pbar = tqdm(total= lenth)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = 0
    ys, preds = torch.tensor([]).to(device), torch.tensor([]).to(device)

    loss_op = torch.nn.BCELoss()

    for g_train in train_list:

        x = g_train.ndata['feat'].to(device)
        y = g_train.ndata['label'].to(device)
        edge_index = torch.cat((g_train.edges()[0].reshape(1, g_train.edges()[0].shape[0]),
                                g_train.edges()[1].reshape(1, g_train.edges()[1].shape[0])), 0).to(torch.int64).to(
            device)
        # edge_index = to_undirected(edge_index)
        edge_index = add_self_loops(edge_index)[0]
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = loss_op(out.float(), y.squeeze().float())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        ys = torch.cat([ys, y])
        preds = torch.cat([preds, out])

        pbar.update(1)

    pbar.close()
    ys, preds = ys.squeeze(), preds.squeeze()
    ys, preds = ys.cpu().detach().numpy(), preds.cpu().detach().numpy()
    return total_loss / lenth, roc_auc_score(ys, preds), ys, preds

@torch.no_grad()
def validation(g_valid, model):
    model.eval()
    loss_op = torch.nn.BCELoss()
    ys, preds = torch.tensor([]).to(device), torch.tensor([]).to(device)

    x = g_valid.ndata['feat'].to(device)
    y = g_valid.ndata['label'].to(device)
    edge_index = torch.cat((g_valid.edges()[0].reshape(1, g_valid.edges()[0].shape[0]),
                            g_valid.edges()[1].reshape(1, g_valid.edges()[1].shape[0])), 0).to(torch.int64).to(device)
    # edge_index = to_undirected(edge_index)
    edge_index = add_self_loops(edge_index)[0]
    out = model(x, edge_index)
    loss = loss_op(out.float(), y.squeeze().float())
    ys = torch.cat([ys, y])
    preds = torch.cat([preds, out])

    ys, preds = ys.squeeze(), preds.squeeze()
    ys, preds = ys.cpu().detach().numpy(), preds.cpu().detach().numpy()

    return roc_auc_score(ys, preds), ys, preds, loss

@torch.no_grad()
def test(g_test, model):
    model.eval()
    ys, preds = torch.tensor([]).to(device), torch.tensor([]).to(device)

    x = g_test.ndata['feat'].to(device)
    y = g_test.ndata['label'].to(device)
    edge_index = torch.cat((g_test.edges()[0].reshape(1, g_test.edges()[0].shape[0]),
                            g_test.edges()[1].reshape(1, g_test.edges()[1].shape[0])), 0).to(torch.int64).to(device)
    print(edge_index.shape)
    # edge_index = to_undirected(edge_index)
    # edge_index = add_self_loops(edge_index)[0]
    out = model(x, edge_index)
    ys = torch.cat([ys, y])
    preds = torch.cat([preds, out])

    ys, preds = ys.squeeze(), preds.squeeze()
    ys, preds = ys.cpu().detach().numpy(), preds.cpu().detach().numpy()

    return roc_auc_score(ys, preds), ys, preds

epoch_fig=[]
loss_fig=[]
train_auc_fig=[]
test_auc_fig=[]

best_loss = 100
best_epoch = 0
best_valid_auc = 0.5

for epoch in range(1, args.epoch):
    random.shuffle(data)
    loss, train_auc, train_y_train, train_y_pred = train(epoch, data, model)
    valid_auc, valid_y_test, valid_y_pred, valid_loss = validation(g_valid, model)
    test_auc, test_y_test, test_y_pred = test(g_test, model)

    print('\n')
    print('Epoch: {:02d}, Train_Loss: {:.4f}, Train_AUC: {:.4f} , '
          'Valid_Loss: {:.4f}, Valid_AUC: {:.4f}, Test_AUC: {:.4f} '.format(
        epoch, loss, train_auc, valid_loss, valid_auc, test_auc))
    print('\n')

    if valid_loss < best_loss:

        best_loss = valid_loss
        best_epoch = epoch
        best_valid_auc = valid_auc
        checkpoint = copy.deepcopy(model)
        es = 0

    else:
        es += 1
        if es > 9:
            break

print('best_epoch:{:.4f}, best_valid_loss:{:.4f}, best_valid_auc:{:.4f}' .format(best_epoch, best_loss, best_valid_auc))

endtime = time.time()
dtime = endtime - starttime
print("time：%.8s s" % dtime)

model = checkpoint
# test_auc, test_y_test, test_y_pred = test()
# print(test_auc)
torch.save(model, 'model/NHEK_model/chr_'+ str(args.test_chr) +'.pt')
