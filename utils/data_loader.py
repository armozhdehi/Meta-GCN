import pandas as pd
from utils.pre_processing import fix_imbalance_ratio
import networkx as nx

# Generate the Pima Indian Diabetes (Diabetes) graph through the method used in : https://arxiv.org/abs/2103.00221
def data_loader_diabetes(args):
    diabetes_df = pd.read_csv('./data/diabetes.csv')
    # print(diabetes_df.head(10))
    dataset_length = diabetes_df.values.shape[0]
    diabetes_n_features = diabetes_df.values.shape[1] - 2
    diabetes_Y_np = diabetes_df.drop(columns=('Glucose')).values[:, -1].unique()
    diabetes_n_labels = len(pd.DataFrame(diabetes_Y_np, columns=['labels']).labels.unique())
    # print(diabetes_Y_df)

    diabetes_train_X_df = diabetes_df.drop(columns=('Glucose')).values[:int(dataset_length * args.train_ratio), :-1]
    diabetes_train_Y_df = diabetes_df.values[:int(dataset_length * args.train_ratio), -1]
    diabetes_val_X_df = diabetes_df.drop(columns=('Glucose')).values[int(dataset_length * args.train_ratio):int(dataset_length * (1 - args.test_ratio)), :-1]
    diabetes_val_Y_df = diabetes_df.values[int(dataset_length * args.train_ratio):int(dataset_length * (1 - args.test_ratio)), -1]
    diabetes_test_X_df = diabetes_df.drop(columns=('Glucose')).values[int(dataset_length * (1 - args.test_ratio)):, :-1]
    diabetes_test_Y_df = diabetes_df.values[int(dataset_length * (1 - args.test_ratio)):, -1]
    diabetes_train_idx = list(range(int(dataset_length * args.train_ratio)))
    diabetes_val_idx = list(range(int(dataset_length * args.train_ratio), int(dataset_length * (1 - args.test_ratio))))
    diabetes_test_idx = list(range(int(dataset_length * (1 - args.test_ratio)), dataset_length))
    diabetes_train_X_df, diabetes_train_Y_df, diabetes_train_idx, diabetes_val_idx, diabetes_test_idx = fix_imbalance_ratio(args.imbalance_ratio, \
        diabetes_train_X_df, diabetes_train_Y_df, diabetes_train_idx, diabetes_val_idx, diabetes_test_idx)
    G = nx.Graph()
    gam = 4
    for patient_id, row_vals in diabetes_df.iterrows():
        G.add_node(str(patient_id), 
        pregnancies = row_vals[0], 
        # glucose = row_vals[1], 
        bloodpressure = row_vals[2], 
        skinthickness = row_vals[3], 
        insulin = row_vals[4], 
        bmi = row_vals[5], 
        diabetespedigreefunction = row_vals[6], 
        age = row_vals[7]) 
        # , outcome = row_vals[8])
    # Two loops because of the order problem of NetworkX
    for patient_id, row_vals in diabetes_df.iterrows():
        for other_patient_id in range(patient_id + 1, diabetes_df.shape[0]):
            if abs(diabetes_df.iloc[[other_patient_id], 1].values[0] - diabetes_df.iloc[[patient_id], 1].values[0]) < gam:
                G.add_edge(*(str(patient_id), str(other_patient_id)))
    diabetes_adj_mtx = nx.to_numpy_matrix(G)
    # print(diabetes_adj_mtx)
    # print(G)
    # pd.DataFrame(diabetes_adj_mtx).to_csv("data/diabetes_adj_mtx.csv")
    return diabetes_adj_mtx, diabetes_train_X_df, diabetes_train_Y_df, diabetes_val_X_df, diabetes_val_Y_df, diabetes_test_X_df, diabetes_test_Y_df \
        , diabetes_train_idx, diabetes_val_idx, diabetes_test_idx, diabetes_n_features

# Haberman’s survival (Haberman) graph through the method used in : https://arxiv.org/abs/2103.00221
def data_loader_haberman(args):
    haberman_df = pd.read_csv('./data/haberman.csv')
    # print(diabetes_df.head(10))
    dataset_length = haberman_df.values.shape[0]
    haberman_n_features = haberman_df.values.shape[1] - 2
    haberman_train_X_df = haberman_df.drop(columns=('Lymph Nodes')).values[:int(dataset_length * args.train_ratio), :-1]
    haberman_train_Y_df = haberman_df.values[:int(dataset_length * args.train_ratio), -1]
    haberman_val_X_df = haberman_df.drop(columns=('Lymph Nodes')).values[int(dataset_length * args.train_ratio):int(dataset_length * (1 - args.train_ratio)), :-1]
    haberman_val_Y_df = haberman_df.values[int(dataset_length * args.train_ratio):int(dataset_length * (1 - args.train_ratio)), -1]
    haberman_test_X_df = haberman_df.drop(columns=('Lymph Nodes')).values[int(dataset_length * (1 - args.train_ratio)):, :-1]
    haberman_test_Y_df = haberman_df.values[int(dataset_length * (1 - args.train_ratio)):, -1]
    haberman_train_idx = list(range(int(dataset_length * args.train_ratio)))
    haberman_val_idx = list(range(int(dataset_length * args.train_ratio), int(dataset_length * (1 - args.train_ratio))))
    haberman_test_idx = list(range(int(dataset_length * (1 - args.train_ratio)), dataset_length))
    haberman_train_X_df, haberman_train_Y_df, haberman_train_idx, haberman_val_idx, haberman_test_idx = fix_imbalance_ratio(args.imbalance_ratio, \
        haberman_train_X_df, haberman_train_Y_df, haberman_train_idx, haberman_val_idx, haberman_test_idx)
    G = nx.Graph()
    gam = 2
    for patient_id, row_vals in haberman_df.iterrows():
        G.add_node(str(patient_id),
            age = row_vals[0],
            operation_year = row_vals[1],
            lymph_nodes = row_vals[2],
        )
        # survival = row_vals[3])
    for patient_id, row_vals in haberman_df.iterrows():
        for other_patient_id in range(patient_id + 1, haberman_df.shape[0]):
            if abs(haberman_df.iloc[[other_patient_id], 2].values[0] - haberman_df.iloc[[patient_id], 2].values[0]) < gam:
                G.add_edge(*(str(patient_id), str(other_patient_id)))
    haberman_adj_mtx = nx.to_numpy_matrix(G)
    # print(haberman_adj_mtx)
    # print(G)
    # pd.DataFrame(diabetes_adj_mtx).to_csv("data/haberman_adj_mtx.csv")

    return haberman_adj_mtx, haberman_n_features, haberman_n_features, haberman_train_X_df, haberman_val_X_df, haberman_test_X_df, haberman_train_Y_df \
        , haberman_val_Y_df, haberman_test_Y_df, haberman_train_idx, haberman_val_idx, haberman_test_idx