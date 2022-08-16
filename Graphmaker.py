import numpy as np
import torch
import dgl
import pandas as pd
import argparse

# NHEK
for i in range(1, 23):
    triplet = np.loadtxt('/data/wangxu/NHEK/adjancy_matrix_5000_2/chr' + str(i) + ".adjancy.triplet.txt")
    index = triplet[:,2] < 10
    triplet = triplet[index]

    edge_1 = np.hstack((triplet[:, 0], triplet[:, 1])) - 1
    edge_2 = np.hstack((triplet[:, 1], triplet[:, 0])) - 1

    # graph maker
    u, v = torch.tensor(edge_1).int(), torch.tensor(edge_2).int()
    g = dgl.graph((u, v))

    a = g.num_nodes()
    b = g.num_edges()

    # edge maker
    edata = np.hstack((triplet[:, 2], triplet[:, 2]))
    g.edata["edge_feature"] = torch.tensor(edata)

    # seq_feature maker
    df_seq_mer_3 = pd.read_table('/data/wangxu/NHEK/Node_feature_5000_2/DNA_seq_mer_3/chr' + str(i) + '.3.txt', header=None,
                                 low_memory=False).iloc[1:, 1:]
    seq_mer_3 = (df_seq_mer_3.values).astype(np.float64)
    df_seq_mer_4 = pd.read_table('/data/wangxu/NHEK/Node_feature_5000_2/DNA_seq_mer_4/chr' + str(i) + '.4.txt', header=None,
                                 low_memory=False).iloc[1:, 1:]
    seq_mer_4 = (df_seq_mer_4.values).astype(np.float64)
    df_seq_mer_5 = pd.read_table('/data/wangxu/NHEK/Node_feature_5000_2/DNA_seq_mer_5/chr' + str(i) + '.5.txt', header=None,
                                 low_memory=False).iloc[1:, 1:]
    seq_mer_5 = (df_seq_mer_5.values).astype(np.float64)
    x = torch.tensor(np.concatenate((seq_mer_3, seq_mer_4, seq_mer_5), axis=1))
    g.ndata["seq_feature"] = x
#
#     # label maker
    df_lable = pd.read_table('/data/wangxu/NHEK/label_5000_2/chr' + str(i) + '.label.txt', header=None).iloc[:, 1:]
    y = torch.tensor(df_lable.values)
    y = torch.squeeze(y)

    g.ndata["label"] = y

    # id maker
    g.ndata['node_id'] = g.nodes().reshape(g.nodes().shape[0], 1)

    # deg maker
    g.ndata['deg'] = g.out_degrees().float().clamp(min=1)

    dgl.save_graphs("/data/NHEK_dgl_data/chr_" + str(i) + ".dgl", g)

for i in range(1, 2):
    triplet = np.loadtxt('/data/wangxu/NHEK/adjancy_matrix_5000_2/chrX.adjancy.triplet.txt')
    index = triplet[:, 2] < 10
    triplet = triplet[index]

    edge_1 = np.hstack((triplet[:, 0], triplet[:, 1])) - 1
    edge_2 = np.hstack((triplet[:, 1], triplet[:, 0])) - 1

    # graph maker
    u, v = torch.tensor(edge_1).int(), torch.tensor(edge_2).int()
    g = dgl.graph((u, v))

    a = g.num_nodes()
    b = g.num_edges()

    # edge maker
    edata = np.hstack((triplet[:, 2], triplet[:, 2]))
    g.edata["edge_feature"] = torch.tensor(edata)

    # seq_feature maker
    df_seq_mer_3 = pd.read_table('/data/wangxu/NHEK/Node_feature_5000_2/DNA_seq_mer_3/chrX.3.txt', header=None,
                                 low_memory=False).iloc[1:, 1:]
    seq_mer_3 = (df_seq_mer_3.values).astype(np.float64)
    df_seq_mer_4 = pd.read_table('/data/wangxu/NHEK/Node_feature_5000_2/DNA_seq_mer_4/chrX.4.txt', header=None,
                                 low_memory=False).iloc[1:, 1:]
    seq_mer_4 = (df_seq_mer_4.values).astype(np.float64)
    df_seq_mer_5 = pd.read_table('/data/wangxu/NHEK/Node_feature_5000_2/DNA_seq_mer_5/chrX.5.txt', header=None,
                                 low_memory=False).iloc[1:, 1:]
    seq_mer_5 = (df_seq_mer_5.values).astype(np.float64)
    x = torch.tensor(np.concatenate((seq_mer_3, seq_mer_4, seq_mer_5), axis=1))
    g.ndata["seq_feature"] = x

    # label maker
    df_lable = pd.read_table('/data/wangxu/NHEK/label_5000_2/chrX.label.txt', header=None).iloc[:, 1:]
    y = torch.tensor(df_lable.values)
    y = torch.squeeze(y)
    g.ndata["label"] = y

    # id maker
    g.ndata['node_id'] = g.nodes().reshape(g.nodes().shape[0], 1)

    # deg maker
    g.ndata['deg'] = g.out_degrees().float().clamp(min=1)

    dgl.save_graphs("/data/wangxu/NHEK_dgl_data/chr_23.dgl", g)


#K562
for i in range(1, 23):
    triplet = np.loadtxt('/data/wangxu/K562/adjancy_matrix_5000_2/chr' + str(i) + ".adjancy.triplet.txt")
    index = triplet[:,2] < 10
    triplet = triplet[index]

    edge_1 = np.hstack((triplet[:, 0], triplet[:, 1])) - 1
    edge_2 = np.hstack((triplet[:, 1], triplet[:, 0])) - 1

    # graph maker
    u, v = torch.tensor(edge_1).int(), torch.tensor(edge_2).int()
    g = dgl.graph((u, v))

    a = g.num_nodes()
    b = g.num_edges()

    # edge maker
    edata = np.hstack((triplet[:, 2], triplet[:, 2]))
    g.edata["edge_feature"] = torch.tensor(edata)

    # seq_feature maker
    df_seq_mer_3 = pd.read_table('/data/wangxu/K562/Node_feature_5000_2/DNA_seq_mer_3/chr' + str(i) + '.3.txt', header=None,
                                 low_memory=False).iloc[1:, 1:]
    seq_mer_3 = (df_seq_mer_3.values).astype(np.float64)
    df_seq_mer_4 = pd.read_table('/data/wangxu/K562/Node_feature_5000_2/DNA_seq_mer_4/chr' + str(i) + '.4.txt', header=None,
                                 low_memory=False).iloc[1:, 1:]
    seq_mer_4 = (df_seq_mer_4.values).astype(np.float64)
    df_seq_mer_5 = pd.read_table('/data/wangxu/K562/Node_feature_5000_2/DNA_seq_mer_5/chr' + str(i) + '.5.txt', header=None,
                                 low_memory=False).iloc[1:, 1:]
    seq_mer_5 = (df_seq_mer_5.values).astype(np.float64)
    x = torch.tensor(np.concatenate((seq_mer_3, seq_mer_4, seq_mer_5), axis=1))
    g.ndata["seq_feature"] = x
#
#     # label maker
    df_lable = pd.read_table('/data/wangxu/K562/label_5000_2/chr' + str(i) + '.label.txt', header=None).iloc[:, 1:]
    y = torch.tensor(df_lable.values)
    y = torch.squeeze(y)

    g.ndata["label"] = y

    # id maker
    g.ndata['node_id'] = g.nodes().reshape(g.nodes().shape[0], 1)

    # deg maker
    g.ndata['deg'] = g.out_degrees().float().clamp(min=1)

    dgl.save_graphs("/data/wangxu/K562_dgl_data/chr_" + str(i) + ".dgl", g)

#
for i in range(1, 2):
    triplet = np.loadtxt('/data/wangxu/K562/adjancy_matrix_5000_2/chrX.adjancy.triplet.txt')
    index = triplet[:, 2] < 10
    triplet = triplet[index]

    edge_1 = np.hstack((triplet[:, 0], triplet[:, 1])) - 1
    edge_2 = np.hstack((triplet[:, 1], triplet[:, 0])) - 1

    # graph maker
    u, v = torch.tensor(edge_1).int(), torch.tensor(edge_2).int()
    g = dgl.graph((u, v))

    a = g.num_nodes()
    b = g.num_edges()

    # edge maker
    edata = np.hstack((triplet[:, 2], triplet[:, 2]))
    g.edata["edge_feature"] = torch.tensor(edata)

    # seq_feature maker
    df_seq_mer_3 = pd.read_table('/data/wangxu/K562/Node_feature_5000_2/DNA_seq_mer_3/chrX.3.txt', header=None,
                                 low_memory=False).iloc[1:, 1:]
    seq_mer_3 = (df_seq_mer_3.values).astype(np.float64)
    df_seq_mer_4 = pd.read_table('/data/wangxu/K562/Node_feature_5000_2/DNA_seq_mer_4/chrX.4.txt', header=None,
                                 low_memory=False).iloc[1:, 1:]
    seq_mer_4 = (df_seq_mer_4.values).astype(np.float64)
    df_seq_mer_5 = pd.read_table('/data/wangxu/K562/Node_feature_5000_2/DNA_seq_mer_5/chrX.5.txt', header=None,
                                 low_memory=False).iloc[1:, 1:]
    seq_mer_5 = (df_seq_mer_5.values).astype(np.float64)
    x = torch.tensor(np.concatenate((seq_mer_3, seq_mer_4, seq_mer_5), axis=1))
    g.ndata["seq_feature"] = x
#
    # label maker
    df_lable = pd.read_table('/data/wangxu/K562/label_5000_2/chrX.label.txt', header=None).iloc[:, 1:]
    y = torch.tensor(df_lable.values)
    y = torch.squeeze(y)
    g.ndata["label"] = y

    # id maker
    g.ndata['node_id'] = g.nodes().reshape(g.nodes().shape[0], 1)

    # deg maker
    g.ndata['deg'] = g.out_degrees().float().clamp(min=1)

    dgl.save_graphs("/data/wangxu/K562_dgl_data/chr_23.dgl", g)