import pandas as pd
# from utils.pre_processing import fix_imbalance_ratio
from utils.pre_processing import fix_imbalance_ratio, normalize, sparse_mx_to_torch_sparse_tensor, print_edges_num, fix_imbalance_ratio2, fix_imbalance_ratio_cora
import numpy as np
import networkx as nx
import torch
import argparse
import scipy.sparse as sp
import numpy as np
import torch
import ipdb
from scipy.io import loadmat
import networkx as nx
import multiprocessing as mp
import torch.nn.functional as F
from functools import partial
import random
from sklearn.metrics import roc_auc_score, f1_score
from copy import deepcopy
from scipy.spatial.distance import pdist,squareform
from torch_geometric.datasets import Planetoid
import torch_geometric.utils.to_dense_adj as to_dense_adj


def data_loader_cora(args, path="data/cora/"):#modified from code: pygcn
    """Load citation network dataset (cora only for now)"""
    #input: idx_features_labels, adj
    #idx,labels are not required to be processed in advance
    #adj: save in the form of edges. idx1 idx2 
    #output: adj, features, labels are all torch.tensor, in the dense form
    #-------------------------------------------------------
    # print('Loading {} dataset...'.format(dataset))
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, "cora"),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = idx_features_labels[:, -1]
    set_labels = set(labels)
    classes_dict = {c: np.arange(len(set_labels))[i] for i, c in enumerate(set_labels)}
    classes_dict = {'Neural_Networks': 0, 'Reinforcement_Learning': 1, 'Probabilistic_Methods': 2, 'Case_Based': 3, 'Theory': 4, 'Rule_Learning': 5, 'Genetic_Algorithms': 6}
    #ipdb.set_trace()
    labels = np.array(list(map(classes_dict.get, labels)))
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int64)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, "cora"),
                                    dtype=np.int64)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int64).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float64)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = normalize(features)
    features = torch.DoubleTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    # print_edges_num(adj.todense(), labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    train_idx = np.array([])
    val_idx = np.array([])
    test_idx = np.array([])
    Y_np = torch.unique(labels)
    labels_df = pd.DataFrame(labels.numpy())
    n_labels = len(Y_np)
    for outcome in Y_np:
        outcome_group = labels_df[labels_df == outcome]
        first = int(outcome_group.shape[0] * args.train_ratio)
        last = int(outcome_group.shape[0] * (1 - args.test_ratio))
        train_idx = np.append(train_idx, outcome_group[: first].index)
        val_idx = np.append(val_idx, outcome_group[first: last].index)
        test_idx = np.append(test_idx, outcome_group[last:].index)

    #adj = torch.FloatTensor(np.array(adj.todense()))
    n_features = features.shape[1]
    features, labels, train_idx, val_idx ,test_idx = fix_imbalance_ratio_cora(args.imbalance_ratio, \
        labels, features, train_idx, val_idx, test_idx)
    return adj, labels, features, \
        train_idx, val_idx, test_idx, n_features


# Generate the Pima Indian Diabetes (Diabetes) graph through the method used in : https://arxiv.org/abs/2103.00221
def data_loader_diabetes(args):
    diabetes_df = pd.read_csv('./data/diabetes.csv')
    # print(diabetes_df.head(10))
    dataset_length = diabetes_df.values.shape[0]
    diabetes_n_features = diabetes_df.values.shape[1] - 2
    # diabetes_Y_np = diabetes_df.drop(columns=('Glucose')).unique()
    diabetes_labels_df = diabetes_df.drop(columns=('Glucose')).values[:, -1]
    diabetes_features_df = diabetes_df.drop(columns=('Glucose')).values[:, :-1]
    # diabetes_n_labels = len(pd.DataFrame(diabetes_Y_np, columns=['labels']).labels.unique())
    diabetes_train_idx = np.array([])
    diabetes_val_idx = np.array([])
    diabetes_test_idx = np.array([])
    diabetes_Y_np = diabetes_df['Outcome'].unique()
    diabetes_n_labels = len(diabetes_Y_np)
    for outcome in diabetes_Y_np:
        outcome_group = diabetes_df[diabetes_df['Outcome'] == outcome]
        first = int(outcome_group.shape[0] * args.train_ratio)
        last = int(outcome_group.shape[0] * (1 - args.test_ratio))
        diabetes_train_idx = np.append(diabetes_train_idx, outcome_group[:first].index)
        diabetes_val_idx = np.append(diabetes_val_idx, outcome_group[first:last].index)
        diabetes_test_idx = np.append(diabetes_test_idx, outcome_group[last:].index)
    # print(diabetes_Y_df)
    # diabetes_train_X_df = diabetes_df.drop(columns=('Glucose')).values[:int(dataset_length * args.train_ratio), :-1]
    # diabetes_train_Y_df = diabetes_df.values[:int(dataset_length * args.train_ratio), -1]
    # diabetes_val_X_df = diabetes_df.drop(columns=('Glucose')).values[int(dataset_length * args.train_ratio):int(dataset_length * (1 - args.test_ratio)), :-1]
    # diabetes_val_Y_df = diabetes_df.values[int(dataset_length * args.train_ratio):int(dataset_length * (1 - args.test_ratio)), -1]
    # diabetes_test_X_df = diabetes_df.drop(columns=('Glucose')).values[int(dataset_length * (1 - args.test_ratio)):, :-1]
    # diabetes_test_Y_df = diabetes_df.values[int(dataset_length * (1 - args.test_ratio)):, -1]
    # diabetes_train_idx = list(range(int(dataset_length * args.train_ratio)))
    # diabetes_val_idx = list(range(int(dataset_length * args.train_ratio), int(dataset_length * (1 - args.test_ratio))))
    # diabetes_test_idx = list(range(int(dataset_length * (1 - args.test_ratio)), dataset_length))

    diabetes_features_df, diabetes_labels_df, diabetes_train_idx, diabetes_val_idx ,diabetes_val_idx = fix_imbalance_ratio(args.imbalance_ratio, \
        diabetes_labels_df, diabetes_features_df, diabetes_train_idx, diabetes_val_idx, diabetes_test_idx)

    # diabetes_features_df, diabetes_labels_df, diabetes_train_idx, diabetes_val_idx ,diabetes_val_idx = fix_imbalance_ratio2(args.imbalance_ratio, \
    #     diabetes_labels_df, diabetes_features_df, diabetes_train_idx, diabetes_val_idx, diabetes_test_idx)

    G = nx.Graph()
    gam = 4
    # diabetes_df = [diabetes_labels_df, iabetes_features_df]
    diabetes_features_df = normalize(diabetes_features_df)
    for patient_id, row_vals in diabetes_df.iterrows():
        # print(patient_id, row_vals)
        G.add_node(str(patient_id), 
        pregnancies = row_vals['Pregnancies'], 
        # glucose = row_vals[1], 
        bloodpressure = row_vals['BloodPressure'], 
        skinthickness = row_vals['SkinThickness'], 
        insulin = row_vals['Insulin'], 
        bmi = row_vals['BMI'], 
        diabetespedigreefunction = row_vals['DiabetesPedigreeFunction'], 
        age = row_vals['Age']) 
        # , outcome = row_vals[8])
    # Two loops because of the order problem of NetworkX
    for patient_id, row_vals in diabetes_df.iterrows():
        for other_patient_id in range(patient_id + 1, diabetes_df.shape[0]):
            if abs(diabetes_df['Glucose'][other_patient_id] - diabetes_df['Glucose'][patient_id]) < gam:
                G.add_edge(*(str(patient_id), str(other_patient_id)))
    diabetes_adj_mtx = nx.to_numpy_matrix(G)
    # print(diabetes_adj_mtx)
    # print(G)
    # pd.DataFrame(diabetes_adj_mtx).to_csv("data/diabetes_adj_mtx.csv")
    return diabetes_adj_mtx, diabetes_labels_df, diabetes_features_df, \
        diabetes_train_idx, diabetes_val_idx, diabetes_test_idx, diabetes_n_features

# Haberman’s survival (Haberman) graph through the method used in : https://arxiv.org/abs/2103.00221
def data_loader_haberman(args):
    # haberman_df = pd.read_csv('./data/haberman.csv')
    # # print(diabetes_df.head(10))
    # # dataset_length = haberman_df.values.shape[0]
    # # haberman_n_features = haberman_df.values.shape[1] - 2
    # # haberman_train_X_df = haberman_df.drop(columns=('Lymph Nodes')).values[:int(dataset_length * args.train_ratio), :-1]
    # # haberman_train_Y_df = haberman_df.values[:int(dataset_length * args.train_ratio), -1]
    # # haberman_val_X_df = haberman_df.drop(columns=('Lymph Nodes')).values[int(dataset_length * args.train_ratio):int(dataset_length * (1 - args.train_ratio)), :-1]
    # # haberman_val_Y_df = haberman_df.values[int(dataset_length * args.train_ratio):int(dataset_length * (1 - args.train_ratio)), -1]
    # # haberman_test_X_df = haberman_df.drop(columns=('Lymph Nodes')).values[int(dataset_length * (1 - args.train_ratio)):, :-1]
    # # haberman_test_Y_df = haberman_df.values[int(dataset_length * (1 - args.train_ratio)):, -1]
    # # haberman_train_idx = list(range(int(dataset_length * args.train_ratio)))
    # # haberman_val_idx = list(range(int(dataset_length * args.train_ratio), int(dataset_length * (1 - args.train_ratio))))
    # # haberman_test_idx = list(range(int(dataset_length * (1 - args.train_ratio)), dataset_length))
    # # haberman_train_X_df, haberman_train_Y_df, haberman_train_idx, haberman_val_idx, haberman_test_idx = fix_imbalance_ratio(args.imbalance_ratio, \
    # #     haberman_train_X_df, haberman_train_Y_df, haberman_train_idx, haberman_val_idx, haberman_test_idx)
    # dataset_length = haberman_df.values.shape[0]
    # haberman_n_features = haberman_df.values.shape[1] - 2
    # # diabetes_Y_np = diabetes_df.drop(columns=('Glucose')).unique()
    # haberman_np = haberman_df.drop(columns=('Glucose')).values[:, -1]
    # # diabetes_n_labels = len(pd.DataFrame(diabetes_Y_np, columns=['labels']).labels.unique())

    # haberman_Y_np = haberman_df['Outcome'].unique()
    # haberman_n_labels = len(haberman_Y_np)
    # for outcome in haberman_Y_np:
    #     outcome_group = haberman_df[haberman_df['Outcome'] == outcome]
    #     haberman_train_idx = outcome_group[:int(len(outcome_group) * args.train_ratio)].index 
    #     haberman_val_idx = outcome_group[int(len(outcome_group) * args.train_ratio):int(len(outcome_group) * (1 - args.test_ratio))].index 
    #     haberman_test_idx = outcome_group[:int(len(outcome_group) * (1 - args.test_ratio))].index 
    # G = nx.Graph()
    haberman_df = pd.read_csv('./data/haberman.csv')
    # print(diabetes_df.head(10))
    haberman_length = haberman_df.values.shape[0]
    haberman_n_features = haberman_df.values.shape[1] - 2
    # diabetes_Y_np = diabetes_df.drop(columns=('Glucose')).unique()
    haberman_labels_df = haberman_df.drop(columns=('Lymph Nodes')).values[:, -1]
    haberman_features_df = haberman_df.drop(columns=('Lymph Nodes')).values[:, :-1]
    # diabetes_n_labels = len(pd.DataFrame(diabetes_Y_np, columns=['labels']).labels.unique())
    haberman_train_idx = np.array([])
    haberman_val_idx = np.array([])
    haberman_test_idx = np.array([])
    haberman_Y_np = haberman_df['Survival'].unique()
    haberman_n_labels = len(haberman_Y_np)
    for outcome in haberman_Y_np:
        outcome_group = haberman_df[haberman_df['Survival'] == outcome]
        first = int(outcome_group.shape[0] * args.train_ratio)
        last = int(outcome_group.shape[0] * (1 - args.test_ratio))
        haberman_train_idx = np.append(haberman_train_idx, outcome_group[:first].index)
        haberman_val_idx = np.append(haberman_val_idx, outcome_group[first:last].index)
        haberman_test_idx = np.append(haberman_test_idx, outcome_group[last:].index)
    # print(diabetes_Y_df)
    # diabetes_train_X_df = diabetes_df.drop(columns=('Glucose')).values[:int(dataset_length * args.train_ratio), :-1]
    # diabetes_train_Y_df = diabetes_df.values[:int(dataset_length * args.train_ratio), -1]
    # diabetes_val_X_df = diabetes_df.drop(columns=('Glucose')).values[int(dataset_length * args.train_ratio):int(dataset_length * (1 - args.test_ratio)), :-1]
    # diabetes_val_Y_df = diabetes_df.values[int(dataset_length * args.train_ratio):int(dataset_length * (1 - args.test_ratio)), -1]
    # diabetes_test_X_df = diabetes_df.drop(columns=('Glucose')).values[int(dataset_length * (1 - args.test_ratio)):, :-1]
    # diabetes_test_Y_df = diabetes_df.values[int(dataset_length * (1 - args.test_ratio)):, -1]
    # diabetes_train_idx = list(range(int(dataset_length * args.train_ratio)))
    # diabetes_val_idx = list(range(int(dataset_length * args.train_ratio), int(dataset_length * (1 - args.test_ratio))))
    # diabetes_test_idx = list(range(int(dataset_length * (1 - args.test_ratio)), dataset_length))

    haberman_features_df, haberman_labels_df, haberman_train_idx, haberman_val_idx ,haberman_val_idx = fix_imbalance_ratio(args.imbalance_ratio, \
        haberman_labels_df, haberman_features_df, haberman_train_idx, haberman_val_idx, haberman_test_idx)

    # haberman_features_df, haberman_labels_df, haberman_train_idx, haberman_val_idx ,haberman_val_idx = fix_imbalance_ratio2(args.imbalance_ratio, \
    #     haberman_labels_df, haberman_features_df, haberman_train_idx, haberman_val_idx, haberman_test_idx)

    G = nx.Graph()
    gam = 2
    # haberman_features_df = normalize(haberman_features_df)
    for patient_id, row_vals in haberman_df.iterrows():
        G.add_node(str(patient_id),
            age = row_vals['Age'],
            operation_year = row_vals['Operation Year']
            # lymph_nodes = row_vals[2],
        )
        # survival = row_vals[3])
    for patient_id, row_vals in haberman_df.iterrows():
        for other_patient_id in range(patient_id + 1, haberman_df.shape[0]):
            if abs(haberman_df['Lymph Nodes'][other_patient_id] - haberman_df['Lymph Nodes'][patient_id]) < gam:
                G.add_edge(*(str(patient_id), str(other_patient_id)))
    haberman_adj_mtx = nx.to_numpy_matrix(G)
    # print(haberman_adj_mtx)
    # print(G)
    # pd.DataFrame(diabetes_adj_mtx).to_csv("data/haberman_adj_mtx.csv")

    return haberman_adj_mtx, haberman_labels_df, haberman_features_df, \
        haberman_train_idx, haberman_val_idx, haberman_test_idx, haberman_n_features

# class Args:
#     weight_decay = 5e-4
#     epochs = 1000
#     learning_rate = 0.01
#     learning_rate_W = 0.01
#     dropout = 0.5
#     dropout_W = 0.5
#     gamma = 1
#     no_cuda = False
#     train_ratio=0.6
#     test_ratio=0.2
#     n_classes = 2
#     seed = 1234
#     dataset = "diabetes"
#     order = 4
#     n_features = 0
#     w_val_size = 10
#     imbalance_ratio = None
#     n_hidden = 0
#     setting = None

# args = Args()

# data_loader_diabetes(args)

def data_loader_cora2(args):
    dataset = Planetoid(root='.', split='full', name="cora")
    data = dataset[0]
    features = data.x
    labels = data.y
    features = features.double()
    labels = labels.long()
    # print_edges_num(adj.todense(), labels)
    adj = to_dense_adj(dataset[0].edge_index)[0]
    train_idx = np.array([])
    val_idx = np.array([])
    test_idx = np.array([])
    Y_np = torch.unique(labels)
    labels_df = pd.DataFrame(labels.numpy())
    n_labels = len(Y_np)
    for outcome in Y_np:
        outcome_group = labels_df[labels_df == outcome]
        first = int(outcome_group.shape[0] * args.train_ratio)
        last = int(outcome_group.shape[0] * (1 - args.test_ratio))
        train_idx = np.append(train_idx, outcome_group[: first].index)
        val_idx = np.append(val_idx, outcome_group[first: last].index)
        test_idx = np.append(test_idx, outcome_group[last:].index)

    #adj = torch.FloatTensor(np.array(adj.todense()))
    n_features = features.shape[1]
    features, labels, train_idx, val_idx ,test_idx = fix_imbalance_ratio_cora(args.imbalance_ratio, \
        labels, features, train_idx, val_idx, test_idx)
    return adj, labels, features, \
        train_idx, val_idx, test_idx, n_features


def data_loader_pubmed(args):
    dataset = Planetoid(root='.', split='full', name="pubmed")
    data = dataset[0]
    features = data.x
    labels = data.y
    features = features.double()
    labels = labels.long()
    # print_edges_num(adj.todense(), labels)
    adj = to_dense_adj(dataset[0].edge_index)[0]
    train_idx = np.array([])
    val_idx = np.array([])
    test_idx = np.array([])
    Y_np = torch.unique(labels)
    labels_df = pd.DataFrame(labels.numpy())
    n_labels = len(Y_np)
    for outcome in Y_np:
        outcome_group = labels_df[labels_df == outcome]
        first = int(outcome_group.shape[0] * args.train_ratio)
        last = int(outcome_group.shape[0] * (1 - args.test_ratio))
        train_idx = np.append(train_idx, outcome_group[: first].index)
        val_idx = np.append(val_idx, outcome_group[first: last].index)
        test_idx = np.append(test_idx, outcome_group[last:].index)

    #adj = torch.FloatTensor(np.array(adj.todense()))
    n_features = features.shape[1]
    features, labels, train_idx, val_idx ,test_idx = fix_imbalance_ratio_cora(args.imbalance_ratio, \
        labels, features, train_idx, val_idx, test_idx)
    return adj, labels, features, \
        train_idx, val_idx, test_idx, n_features

def data_loader_citeseer(args):
    dataset = Planetoid(root='.', split='full', name="CiteSeer")
    data = dataset[0]
    features = data.x
    labels = data.y
    features = features.double()
    labels = labels.long()
    # print_edges_num(adj.todense(), labels)
    adj = to_dense_adj(dataset[0].edge_index)[0]
    train_idx = np.array([])
    val_idx = np.array([])
    test_idx = np.array([])
    Y_np = torch.unique(labels)
    labels_df = pd.DataFrame(labels.numpy())
    n_labels = len(Y_np)
    for outcome in Y_np:
        outcome_group = labels_df[labels_df == outcome]
        first = int(outcome_group.shape[0] * args.train_ratio)
        last = int(outcome_group.shape[0] * (1 - args.test_ratio))
        train_idx = np.append(train_idx, outcome_group[: first].index)
        val_idx = np.append(val_idx, outcome_group[first: last].index)
        test_idx = np.append(test_idx, outcome_group[last:].index)

    #adj = torch.FloatTensor(np.array(adj.todense()))
    n_features = features.shape[1]
    features, labels, train_idx, val_idx ,test_idx = fix_imbalance_ratio_cora(args.imbalance_ratio, \
        labels, features, train_idx, val_idx, test_idx)
    return adj, labels, features, \
        train_idx, val_idx, test_idx, n_features