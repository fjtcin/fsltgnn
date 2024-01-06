import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from sklearn.preprocessing import LabelEncoder


class BinaryLabelEncoder:
    def __init__(self):
        pass

    def fit(self, x):
        pass

    def transform(self, x):
        # return 1 if x is negative, 0 otherwise
        return (x < 0).astype(int)


def preprocess_data(dataset_name, node_feat_dim, edge_feat_dim):
    Path("processed_data/{}/".format(dataset_name)).mkdir(parents=True, exist_ok=True)
    PATH = 'DG_data/{}/soc-sign-{}.csv'.format(dataset_name, dataset_name)
    OUT_DF = 'processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name)
    OUT_FEAT = 'processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name)
    OUT_NODE_FEAT = 'processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name)

    df = pd.read_csv(PATH, header=None)

    le = LabelEncoder()
    combined = pd.concat([df[0], df[1]])

    # Fit the encoder on the combined data
    le.fit(combined)

    # Transform each column separately
    df[0] = le.transform(df[0]) + 1
    df[1] = le.transform(df[1]) + 1

    le_label = BinaryLabelEncoder()
    le_label.fit(df[2])
    df[2] = le_label.transform(df[2])

    df.sort_values(by=3, inplace=True)

    df.rename(columns={
        0: 'u',
        1: 'i',
        3: 'ts',
        2: 'label'
    }, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['idx'] = df.index + 1

    edge_feats = np.zeros((len(df) + 1, edge_feat_dim))
    node_feats = np.zeros((len(le.classes_) + 1, node_feat_dim))

    print('number of nodes ', node_feats.shape[0] - 1)
    print('number of node features ', node_feats.shape[1])
    print('number of edges ', edge_feats.shape[0] - 1)
    print('number of edge features ', edge_feats.shape[1])

    df.to_csv(OUT_DF)  # edge-list
    np.save(OUT_FEAT, edge_feats)  # edge features
    np.save(OUT_NODE_FEAT, node_feats)  # node features


parser = argparse.ArgumentParser('Interface for preprocessing datasets')
parser.add_argument('--dataset_name', type=str,
                    help='Dataset name', default='bitcoinotc', choices=['bitcoinotc', 'bitcoinalpha'])
parser.add_argument('--node_feat_dim', type=int, default=172)
parser.add_argument('--edge_feat_dim', type=int, default=172)

args = parser.parse_args()

preprocess_data(args.dataset_name, args.node_feat_dim, args.edge_feat_dim)
