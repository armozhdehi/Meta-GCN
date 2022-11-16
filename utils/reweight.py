import numpy as np
import pandas as pd

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