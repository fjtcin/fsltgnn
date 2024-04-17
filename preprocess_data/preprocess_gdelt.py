import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def preprocess_data():
    Path("processed_data/gdelt/").mkdir(parents=True, exist_ok=True)
    GRAPH_PATH = 'DG_data/gdelt/edges.csv'
    EDGE_PATH = 'DG_data/gdelt/edge_features.pt'
    LABEL_PATH = 'DG_data/gdelt/labels.csv'
    NODE_PATH = 'DG_data/gdelt/node_features.pt'
    OUT_DF = 'processed_data/gdelt/ml_gdelt.csv'
    OUT_NODE_DF = 'processed_data/gdelt/ml_gdelt_node.csv'
    OUT_FEAT = 'processed_data/gdelt/ml_gdelt.npy'
    OUT_NODE_FEAT = 'processed_data/gdelt/ml_gdelt_node.npy'

    df = pd.read_csv(GRAPH_PATH, usecols=['src', 'dst', 'time'])
    select = np.arange(0, len(df), 100)
    df = df.iloc[select]
    df = df.reset_index(drop=True)
    edge_feats = torch.load(EDGE_PATH).numpy()
    edge_feats = edge_feats[select]

    df2 = pd.read_csv(LABEL_PATH, usecols=['node', 'time', 'label'])
    select = np.arange(0, len(df2), 100)
    df2 = df2.iloc[select]
    df2 = df2.reset_index(drop=True)

    valid_nodes = df2["node"].unique()
    mask = (df["src"].isin(valid_nodes)) & (df["dst"].isin(valid_nodes))
    df = df[mask]
    df = df.reset_index(drop=True)
    edge_feats = edge_feats[mask]

    unique_nodes = set(df["src"]).union(set(df["dst"]))
    mask = df2["node"].isin(unique_nodes)
    df2 = df2[mask]
    df2 = df2.reset_index(drop=True)
    valid_nodes = df2["node"].unique()

    assert set(valid_nodes) == unique_nodes
    node_feats = torch.load(NODE_PATH).numpy()[valid_nodes]

    le = LabelEncoder()
    le.fit(valid_nodes)

    df['src'] = le.transform(df['src']) + 1
    df['dst'] = le.transform(df['dst']) + 1
    df.rename(columns={
        'src': 'u',
        'dst': 'i',
        'time': 'ts',
    }, inplace=True)
    df['idx'] = df.index + 1

    df2['node'] = le.transform(df2['node']) + 1
    df2.insert(0, 'node2', df2['node'])
    df2.rename(columns={
        'node2': 'u',
        'node': 'i',
        'time': 'ts',
        'label': 'label'
    }, inplace=True)
    df2['idx'] = df2.index + 1

    empty = np.zeros(edge_feats.shape[1])
    edge_feats = np.vstack([empty, edge_feats])
    empty = np.zeros(node_feats.shape[1])
    node_feats = np.vstack([empty, node_feats])

    print('number of nodes ', node_feats.shape[0] - 1)
    print('number of node features ', node_feats.shape[1])
    print('number of edges ', edge_feats.shape[0] - 1)
    print('number of edge features ', edge_feats.shape[1])
    print(df2['label'].value_counts().sort_index())

    df.to_csv(OUT_DF)  # edge-list
    df2.to_csv(OUT_NODE_DF)  # node-list
    np.save(OUT_FEAT, edge_feats)  # edge features
    np.save(OUT_NODE_FEAT, node_feats)  # node features

preprocess_data()
