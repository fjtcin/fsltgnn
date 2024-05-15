import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def print_erroneous_df(edf):
    diff_indices = edf.index[edf.index != edf['ACTIONID']]
    frames = []
    for idx in diff_indices:
        frames.append(edf.loc[idx-1 : idx+1])
    print(pd.concat(frames))


def preprocess_data(node_feat_dim):
    Path("processed_data/moocact/").mkdir(parents=True, exist_ok=True)
    GRAPH_PATH = 'DG_data/moocact/mooc_actions.tsv'
    EDGE_PATH = 'DG_data/moocact/mooc_action_features.tsv'
    LABEL_PATH = 'DG_data/moocact/mooc_action_labels.tsv'
    OUT_DF = 'processed_data/moocact/ml_moocact.csv'
    OUT_FEAT = 'processed_data/moocact/ml_moocact.npy'
    OUT_NODE_FEAT = 'processed_data/moocact/ml_moocact_node.npy'

    df = pd.read_csv(GRAPH_PATH, sep='\t')
    df2 = pd.read_csv(EDGE_PATH, sep='\t')
    df3 = pd.read_csv(LABEL_PATH, sep='\t')
    assert (df.index == df['ACTIONID']).all() and (df2.index == df2['ACTIONID']).all()
    # print_erroneous_df(df3)

    df['USERID'] += 1
    df['TARGETID'] += df['USERID'].max() + 1

    assert df['TIMESTAMP'].is_monotonic_increasing

    df.rename(columns={
        'USERID': 'u',
        'TARGETID': 'i',
        'TIMESTAMP': 'ts'
    }, inplace=True)
    df.drop(columns='ACTIONID', inplace=True)
    df['label'] = df3['LABEL']
    df['idx'] = df.index + 1

    edge_feats = df2.drop(columns='ACTIONID').values
    empty = np.zeros(edge_feats.shape[1])
    edge_feats = np.vstack([empty, edge_feats])
    node_feats = np.zeros((df['i'].max() + 1, node_feat_dim))

    print('number of nodes ', node_feats.shape[0] - 1)
    print('number of node features ', node_feats.shape[1])
    print('number of edges ', edge_feats.shape[0] - 1)
    print('number of edge features ', edge_feats.shape[1])
    print(df['label'].value_counts().sort_index())

    df.to_csv(OUT_DF)  # edge-list
    np.save(OUT_FEAT, edge_feats)  # edge features
    np.save(OUT_NODE_FEAT, node_feats)  # node features


parser = argparse.ArgumentParser('Interface for preprocessing datasets')
parser.add_argument('--node_feat_dim', type=int, default=172)

args = parser.parse_args()

preprocess_data(args.node_feat_dim)
