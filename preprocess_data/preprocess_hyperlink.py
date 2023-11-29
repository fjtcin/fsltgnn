import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
# from sklearn.decomposition import PCA


def preprocess_data():
    Path("processed_data/hyperlink/").mkdir(parents=True, exist_ok=True)
    EDGE_PATH = 'DG_data/hyperlink/soc-redditHyperlinks-body.tsv'
    NODE_PATH = 'DG_data/hyperlink/web-redditEmbeddings-subreddits.csv'
    OUT_DF = 'processed_data/hyperlink/ml_hyperlink.csv'
    OUT_FEAT = 'processed_data/hyperlink/ml_hyperlink.npy'
    OUT_NODE_FEAT = 'processed_data/hyperlink/ml_hyperlink_node.npy'

    df = pd.read_csv(EDGE_PATH, sep='\t')

    le = LabelEncoder()
    combined = pd.concat([df['SOURCE_SUBREDDIT'], df['TARGET_SUBREDDIT']])

    # Fit the encoder on the combined data
    le.fit(combined)

    # Transform each column separately
    df['SOURCE_SUBREDDIT'] = le.transform(df['SOURCE_SUBREDDIT']) + 1
    df['TARGET_SUBREDDIT'] = le.transform(df['TARGET_SUBREDDIT']) + 1

    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP']).apply(lambda x: int(x.timestamp()))
    df['LINK_SENTIMENT'].replace(-1, 0, inplace=True)

    df.sort_values(by='TIMESTAMP', inplace=True)

    edge_feats = df['PROPERTIES'].str.split(',', expand=True).astype(float).values

    df.rename(columns={
        'SOURCE_SUBREDDIT': 'u',
        'TARGET_SUBREDDIT': 'i',
        'TIMESTAMP': 'ts',
        'LINK_SENTIMENT': 'label'
    }, inplace=True)
    df = df[['u', 'i', 'ts', 'label']]
    df.reset_index(drop=True, inplace=True)
    df['idx'] = df.index + 1

    # edge feature for zero index, which is not used (since edge id starts from 1)
    empty = np.zeros(edge_feats.shape[1])[np.newaxis, :]
    # Stack arrays in sequence vertically(row wise),
    edge_feats = np.vstack([empty, edge_feats])

    df2 = pd.read_csv(NODE_PATH, header=None)

    known_subreddits = le.classes_
    # Filter df2 to keep only the rows with 'SUBREDDIT' known to the LabelEncoder
    df2 = df2[df2[0].isin(known_subreddits)]
    df2[0] = le.transform(df2[0]) + 1

    df2_array = df2.values
    raw_feats = df2_array[:, 1:]
    # pca = PCA(n_components=172)
    # raw_feats = pca.fit_transform(raw_feats)

    node_feats = np.zeros((len(le.classes_) + 1, raw_feats.shape[1]))
    node_feats[df2_array[:, 0].astype(int), :] = raw_feats[:, :]

    # encoded_labels = np.array([1, 2, 3, 4, 35775, 35776])
    # original_values = le.inverse_transform(encoded_labels-1)
    # print(original_values)
    # print(node_feats[encoded_labels, :])

    # Count the number of feature vectors that are all zeros (i.e., non-existent subreddits)
    non_existent_subreddits = np.count_nonzero(np.all(node_feats == 0, axis=1))
    print("Number of non-existent subreddits: ", non_existent_subreddits)
    print('number of nodes ', node_feats.shape[0] - 1)
    print('number of node features ', node_feats.shape[1])
    print('number of edges ', edge_feats.shape[0] - 1)
    print('number of edge features ', edge_feats.shape[1])

    df.to_csv(OUT_DF)  # edge-list
    np.save(OUT_FEAT, edge_feats)  # edge features
    np.save(OUT_NODE_FEAT, node_feats)  # node features

preprocess_data()
