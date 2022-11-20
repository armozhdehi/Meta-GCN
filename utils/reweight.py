import pandas as pd
import numpy as np
import torch
import random

# Balanced sampling
def balanced_sampling(args, val_size):
    val_total_length = len(args.val_Y)
    labels_df = pd.DataFrame(args.val_Y, columns=['labels'])
    labels = pd.DataFrame(args.val_Y, columns=['labels']).labels.unique()
    val_grouped_df = labels_df.groupby(by=['labels'])['labels'].count()
    val_minority_count = val_grouped_df.min()
    assert val_minority_count > val_size, "The size of minority class is less the the sampling size"
    n_classes = len(val_grouped_df)
    concated_val = np.concatenate([args.val_X, args.val_Y[:, None]], axis=1)
    col_num = concated_val.shape[-1]
    grouped_con_df = pd.DataFrame(pd.DataFrame(concated_val).groupby(by=[col_num - 1]))
    balanced_val_set = pd.DataFrame(columns=[list(range(col_num))]).to_numpy()
    balanced_val_idx = []
    for label_index in range(n_classes):
        temp_df = pd.DataFrame(grouped_con_df.iloc[label_index, 1].sample(n = val_size))
        balanced_val_idx += list(temp_df.index)
        balanced_val_set = pd.DataFrame(np.concatenate([balanced_val_set, temp_df.to_numpy()], axis=0))
    return balanced_val_set, balanced_val_idx

def next(args, features, labels, val_idx, adj_mtx):
    total_data_size = labels.shape[0]
    n_samples = args.val_size
    unique_labels, labels_count = labels[val_idx].reshape(-1).unique(dim=0, return_counts=True)
    sampled_val_idx = None
    for label in unique_labels:
        labels_idxs = (labels == label.item()).nonzero()
        labels_idxs_temp = [idx for idx in labels_idxs if idx in val_idx]
        labels_idxs = torch.tensor(labels_idxs_temp) # labels_idxs = labels_idxs[labels_idxs in val_idx]
        indice = random.sample(range(labels_idxs.shape[0]), n_samples)
        indice = torch.tensor(indice)
        if sampled_val_idx == None:
            sampled_val_idx = labels_idxs[indice]
        else:
            sampled_val_idx = torch.stack((sampled_val_idx, labels_idxs[indice]), 1)
    sampled_val_idx = sampled_val_idx.reshape(-1)
    new_adj_mtx = torch.zeros(sampled_val_idx.shape[0], sampled_val_idx.shape[0])
    for new_idx, idx in enumerate(sampled_val_idx):
        for new_other_idx, other_idx in enumerate(sampled_val_idx):
            if (adj_mtx[idx, other_idx] == 1):
                new_adj_mtx[new_idx, new_other_idx] = 1
    new_features = features[sampled_val_idx]
    new_labels = labels[sampled_val_idx]
    return sampled_val_idx, new_adj_mtx, new_features, new_labels

def next2(args, features, labels, val_idx, adj_mtx):
    total_data_size = labels.shape[0]
    n_samples = args.val_size
    unique_labels, labels_count = labels[val_idx].reshape(-1).unique(dim=0, return_counts=True)
    sampled_val_idx = None
    for label in unique_labels:
        labels_idxs = (labels == label.item()).nonzero()
        labels_idxs_temp = [idx for idx in labels_idxs if idx in val_idx]
        labels_idxs = torch.tensor(labels_idxs_temp) # labels_idxs = labels_idxs[labels_idxs in val_idx]
        indice = random.sample(range(labels_idxs.shape[0]), n_samples)
        indice = torch.tensor(indice)
        if sampled_val_idx == None:
            sampled_val_idx = labels_idxs[indice]
        else:
            sampled_val_idx = torch.stack((sampled_val_idx, labels_idxs[indice]), 1)
    sampled_val_idx = sampled_val_idx.reshape(-1)
    new_adj_mtx = torch.zeros(adj_mtx.shape)
    for idx in sampled_val_idx:
        for other_idx in sampled_val_idx:
            if (adj_mtx[idx, other_idx] == 1):
                new_adj_mtx[idx, other_idx] = 1
    return sampled_val_idx, new_adj_mtx, features, labels
