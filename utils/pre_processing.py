import torch
import numpy as np
import scipy.sparse as sp
import pandas as pd

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def norm_sparse(adj):#normalize a torch dense tensor for GCN, and change it into sparse.
    adj = adj + torch.eye(adj.shape[0]).to(adj)
    rowsum = torch.sum(adj,1)
    r_inv = 1/rowsum
    r_inv[torch.isinf(r_inv)] = 0.
    new_adj = torch.mul(r_inv.reshape(-1,1), adj)

    indices = torch.nonzero(new_adj).t()
    values = new_adj[indices[0], indices[1]] # modify this based on dimensionality
    return torch.sparse.DoubleTensor(indices, values, new_adj.size())

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.DoubleTensor(indices, values, shape)

def find_shown_index(adj, center_ind, steps = 2):
    seen_nodes = {}
    shown_index = []
    if isinstance(center_ind, int):
        center_ind = [center_ind]
    for center in center_ind:
        shown_index.append(center)
        if center not in seen_nodes:
            seen_nodes[center] = 1
    start_point = center_ind
    for step in range(steps):
        new_start_point = []
        candid_point = set(adj[start_point,:].reshape(-1, adj.shape[1]).nonzero()[:,1])
        for i, c_p in enumerate(candid_point):
            if c_p.item() in seen_nodes:
                pass
            else:
                seen_nodes[c_p.item()] = 1
                shown_index.append(c_p.item())
                new_start_point.append(c_p)
        start_point = new_start_point
    return shown_index


# imbalance_ratio = 0.03
# Fixing the train set's imbalance ratio
def fix_imbalance_ratio(imbalance_ratio, train_X, train_Y, train_idx, val_idx, test_idx):
    if imbalance_ratio != None:
        train_total_count = len(train_Y)
        train_labels_df = pd.DataFrame(train_Y, columns=['labels'])
        labels = pd.DataFrame(train_Y, columns=['labels']).labels.unique()
        train_grouped_df = train_labels_df.groupby(by=['labels'])['labels'].count()
        train_minority_count = train_grouped_df.min()
        train_minority_index = train_grouped_df.argmin()
        n_classes = len(train_grouped_df)
        concated_train = np.concatenate([train_X, train_Y[:, None]], axis=1)
        col_num = concated_train.shape[-1]
        grouped_con_df = pd.DataFrame(pd.DataFrame(concated_train).groupby(by=[col_num - 1]))
        current_minority_total_ratio = train_minority_count / train_total_count
        train_minority_label = grouped_con_df.iloc[train_minority_index, 0]
        assert current_minority_total_ratio > imbalance_ratio, "The ratio is below the threshold"
        rest_classes_count = train_total_count - train_minority_count
        reduction = int(imbalance_ratio * rest_classes_count * (imbalance_ratio - 1) + train_minority_count)
        minority_ndarray = grouped_con_df.iloc[train_minority_index, 1]
        drop_indices = np.random.choice(minority_ndarray.index, reduction, replace=False)
        new_train_X = train_X.drop(drop_indices)
        new_train_Y = train_Y.drop(drop_indices)
        train_idx = train_idx[: -reduction]
        val_idx = [x - reduction for x in val_idx]
        test_idx = [x - reduction for x in test_idx]
        return new_train_X, new_train_Y, train_idx, val_idx, test_idx
    return train_X, train_Y, train_idx, val_idx, test_idx