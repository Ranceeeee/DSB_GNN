
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GNNExplainer
import dgl
import time
import numpy as np
from gnn_explainer import GNNExplainer
from torch_geometric.utils import add_self_loops
import argparse
from sklearn import preprocessing
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.utils import to_undirected
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# Settings

parser = argparse.ArgumentParser(description='GNN baselines on pcqm4m with Pytorch Geometrics')
parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
parser.add_argument('--test_chr', type=int, default=1,
                        help='which chromosome used to be test')
parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning_rate')
parser.add_argument('--epoch', type=int, default=20,
                        help='training_epoch')
parser.add_argument('--sleep_time', type=int, default=0,
                        help='wait for run')

args = parser.parse_args()

time.sleep(args.sleep_time)

starttime = time.time()
device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')

test_chr = args.test_chr

train_list = []
print('test_chr:', test_chr)

data = []
for i in range(1, 24):

    g = dgl.load_graphs("/data/K562_dgl_data/chr_" + str(i) + ".dgl")[0][0]
    g = dgl.remove_self_loop(g)
    g.ndata['node_id'] = g.nodes().reshape(g.nodes().shape[0], 1)
    g_density = np.loadtxt('/data/K562/Node_EpiFeature_5000_2/chr' + str(i) + '.density.txt')
    g.ndata['density'] = torch.from_numpy(g_density)
    density_mean_1 = np.nanmean(g_density[:, 0])
    density_mean_2 = np.nanmean(g_density[:, 1])
    nan_index_1 = torch.isnan(g.ndata['density'][:, 0])
    nan_index_2 = torch.isnan(g.ndata['density'][:, 1])
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

for graph in data:
    train_feats = torch.cat([train_feats, graph.ndata['seq_feature']], 0)
    train_density = torch.cat([train_density, graph.ndata['density']], 0)

scaler = preprocessing.StandardScaler().fit(train_feats)
density_scaler = preprocessing.StandardScaler().fit(train_density)

for graph in data:
    graph.ndata['feat'] = torch.from_numpy(scaler.transform(graph.ndata['seq_feature'])).float()
    graph.ndata['density_feat'] = torch.from_numpy(density_scaler.transform(graph.ndata['density'])).float()
    graph.ndata['feat'] = torch.cat((graph.ndata['feat'], graph.ndata['density_feat']), dim=1)

g_valid.ndata['feat'] = torch.from_numpy(scaler.transform(g_valid.ndata['seq_feature'])).float()
g_valid.ndata['density_feat'] = torch.from_numpy(density_scaler.transform(g_valid.ndata['density'])).float()
g_valid.ndata['feat'] = torch.cat((g_valid.ndata['feat'], g_valid.ndata['density_feat']), dim=1)

g_test.ndata['feat'] = torch.from_numpy(scaler.transform(g_test.ndata['seq_feature'])).float()
g_test.ndata['density_feat'] = torch.from_numpy(density_scaler.transform(g_test.ndata['density'])).float()
g_test.ndata['feat'] = torch.cat((g_test.ndata['feat'], g_test.ndata['density_feat']), dim=1)


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

        layer_out = []  # 保存每一层的结果
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

# 加载

model = torch.load('model/K562_model/chr_' + str(args.test_chr) + '.pt', map_location=lambda storage, loc: storage.cuda(args.device))
model = model.to(device)

data = g_test.to(device)
x = data.ndata['feat']
edge_index = torch.cat((g_test.edges()[0].reshape(1, g_test.edges()[0].shape[0]),
                        g_test.edges()[1].reshape(1, g_test.edges()[1].shape[0])), 0).to(torch.int64).to(device)
# edge_index = edge_index + 1
# edge_index = to_undirected(edge_index)
edge_index = add_self_loops(edge_index)[0]

out = model(x, edge_index)
print(out)

ys, preds = torch.tensor([]).to(device), torch.tensor([]).to(device)
y = g_test.ndata['label'].to(device)
ys = torch.cat([ys, y])
preds = torch.cat([preds, out])
ys, preds = ys.squeeze(), preds.squeeze()
ys, preds = ys.cpu().detach().numpy(), preds.cpu().detach().numpy()
print(roc_auc_score(ys, preds))

np.save('prediction/K562/chr_'+ str(args.test_chr) +'_output.npy', preds)
