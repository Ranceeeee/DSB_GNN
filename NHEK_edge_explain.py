import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GNNExplainer
import dgl
import time
import numpy as np
from gnn_explainer import GNNExplainer
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import JumpingKnowledge
import torch.nn as nn
import argparse
from sklearn import preprocessing
import pandas as pd
import openpyxl
from networkx.algorithms import is_isomorphic
import matplotlib.pyplot as plt

plt.switch_backend('agg')

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

    g = dgl.load_graphs("/data/NHEK_dgl_data/chr_" + str(i) + ".dgl")[0][0]
    g = dgl.remove_self_loop(g)
    g.ndata['node_id'] = g.nodes().reshape(g.nodes().shape[0], 1)
    g_density = np.loadtxt('/data/NHEK/Node_EpiFeature_5000_2/chr' + str(i) + '.density.txt')
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

model = torch.load('model/NHEK_model/chr_' + str(args.test_chr) + '.pt', map_location=lambda storage, loc: storage.cuda(args.device))
model = model.to(device)

data = g_test.to(device)
x = data.ndata['feat'].to(device)
y = data.ndata['label'].to(device)

edge_index = torch.cat((g_test.edges()[0].reshape(1,g_test.edges()[0].shape[0]), g_test.edges()[1].reshape(1,g_test.edges()[1].shape[0])),0).to(torch.int64).to(device)
edge_index = add_self_loops(edge_index)[0]
edge_index = remove_self_loops(edge_index)[0]

explainer = GNNExplainer(model, epochs=30, lr=0.005, num_hops=2)

all_edge_top5 = -np.ones((x.shape[0], 10))
connected_edge_top5 = -np.ones((x.shape[0], 10))

all_edge = [['1' for _ in range(14)] for _ in range(x.shape[0] + 10)]
connected_edge = [['1' for _ in range(14)] for _ in range(x.shape[0] + 10)]

all_edge[0][0] = 'node_index'
all_edge[0][1] = 'edges'
all_edge[0][11] = '1-order'
all_edge[0][12] = '2-order'

connected_edge[0][0] = 'node_index'
for i in range(1, 11):
    connected_edge[0][i] = 'Edge'
connected_edge[0][11] = '1-order'
connected_edge[0][12] = '2-order'
connected_edge[0][13] = 'Motif Mode'

# for node_idx in range(0,x.shape[0]):

G_list = []
topology_idx = []
topology_num = []

# for node_idx in range(100):
# part_num = 11
for node_idx in range(0, x.shape[0]):
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
    G, top5_G, connected_edge_list, connected_edge_importance = explainer.visualize_subgraph(node_idx, edge_index, edge_mask, y=y)
    edge_repeat = [node_idx]
    connected_edge_repeat = [node_idx]

    # remove undirected edge
    for i in range(len(connected_edge_list)):
        for j in range(len(connected_edge_list)):
            if (connected_edge_list[i][0], connected_edge_list[i][1]) == (connected_edge_list[j][1], connected_edge_list[j][0]):
                if connected_edge_importance[i] > connected_edge_importance[j]:
                    connected_edge_importance[j] = 0
                else:
                    connected_edge_importance[i] = 0

    if len(connected_edge_importance) != 0:
        connected_edge_importance[0] = connected_edge_importance[0][0]

    connected_edge_importance = torch.tensor(connected_edge_importance)

    if len(list(G.nodes)) == 1:
        all_edge_top5[node_idx][0] = node_idx
        continue

    sorted_edge_mask = torch.sort(edge_mask, descending=True)
    sorted_connected_edge_mask = torch.sort(connected_edge_importance, descending=True)
    all_edge[node_idx + 1][0] = str(node_idx + 1)
    connected_edge[node_idx + 1][0] = str(node_idx + 1)
    hop_1 = ''
    hop_2_list = []
    hop_2 = ''

    all_edge[node_idx + 1][11] = hop_1
    all_edge[node_idx + 1][12] = hop_2

    hop_1 = ''
    hop_1_list = []
    hop_2_list = []
    hop_2 = ''

    for i in range(0, min(10, len(connected_edge_list))):

        index = sorted_connected_edge_mask.indices[i]
        edge_1, edge_2 = connected_edge_list[index][0], connected_edge_list[index][1]

        importance = sorted_connected_edge_mask.values[i].item()
        if importance == 0:
            break

        all_str_edge = "(" + str(min(edge_1, edge_2).item() + 1) + ","
        all_str_edge = all_str_edge + str(max(edge_1, edge_2).item() + 1) + ") " + str(format(importance, '.5f'))
        #
        # all_str_edge = "(" + str(connected_edge_list[i][0].item() + 1) + ',' + str(
        #     connected_edge_list[i][1].item() + 1) + ") imp: " + str(connected_edge_importance[i].item())
        connected_edge[node_idx + 1][i + 1] = all_str_edge
        if edge_1.item() == node_idx and edge_2.item() not in hop_1_list:
            hop_1_list.append(edge_2.item())
            hop_1 = hop_1 + ' ' + str(edge_2.item() + 1)
        elif edge_2.item() == node_idx and edge_1.item() not in hop_1_list:
            hop_1_list.append(edge_1.item())
            hop_1 = hop_1 + ' ' + str(edge_1.item() + 1)
        else:
            if edge_1.item() not in hop_2_list:
                hop_2_list.append(edge_1.item())
                hop_2 = hop_2 + ' ' + str(edge_1.item() + 1)
            elif edge_2.item() not in hop_2_list:
                hop_2_list.append(edge_2.item())
                hop_2 = hop_2 + ' ' + str(edge_2.item() + 1)


    connected_edge[node_idx + 1][11] = hop_1
    connected_edge[node_idx + 1][12] = hop_2

    new_topology = True

    if G_list == []:
        G_list.append(top5_G)
        topology_idx.append(node_idx)
        motif_num = 1
        topology_num.append(1)
        plt.title("motif = " + str(len(topology_num)))
        # plt.show()
        plt.savefig('explainer/NHEK_motif_' + str(args.test_chr) + 'motif_'  + str(len(topology_num))+'.png')
        plt.close()
        continue

    for i in range(len(G_list)):
        if is_isomorphic(G_list[i], top5_G) == True:
            topology_num[i] += 1
            motif_num = i + 1
            new_topology = False
            plt.close()
            break

    if new_topology == True:
        G_list.append(top5_G)
        topology_idx.append(node_idx)
        topology_num.append(1)
        motif_num = len(topology_num)
        plt.title("motif = " + str(len(topology_num)))
        # plt.show()
        plt.savefig('explainer/NHEK_motif_' + str(args.test_chr) + '/motif_'  + str(len(topology_num))+'.png')
        plt.close()

    connected_edge[node_idx + 1][13] = str(motif_num)

wb = openpyxl.Workbook()
ws = wb.active
ws.title = 'connected_edge'
for r in range(len(connected_edge)):
    for c in range(len(connected_edge[0])):
        ws.cell(r + 1, c + 1).value = connected_edge[r][c]
wb.save('explainer/NHEK_chr_' + str(args.test_chr) +'_connected_edge.xlsx')

print('zero_topology_num:%s' % (topology_num))
print('zero_topology_idx:%s' % (topology_idx))

endtime = time.time()
dtime = endtime - starttime
print("time：%.8s s" % dtime)