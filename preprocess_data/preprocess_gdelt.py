import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


SELECTED_LABELS = [2, 12, 26, 56]
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
    edge_feats = torch.load(EDGE_PATH, weights_only=True).numpy()
    edge_feats = edge_feats[select]

    df2 = pd.read_csv(LABEL_PATH, usecols=['node', 'time', 'label'])
    select = np.arange(0, len(df2), 100)
    df2 = df2.iloc[select]
    df2 = df2.reset_index(drop=True)

    df2 = df2[df2['label'].isin(SELECTED_LABELS)]
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
    node_feats = torch.load(NODE_PATH, weights_only=True).numpy()[valid_nodes]

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
    assert df['ts'].is_monotonic_increasing

    df2['node'] = le.transform(df2['node']) + 1
    df2.insert(0, 'node2', df2['node'])
    df2.rename(columns={
        'node2': 'u',
        'node': 'i',
        'time': 'ts',
        'label': 'label'
    }, inplace=True)
    df2['idx'] = df2.index + 1
    assert df2['ts'].is_monotonic_increasing

    le.fit(df2['label'])
    df2['label'] = le.transform(df2['label'])

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

"""
label counts
2     41436
12    28268
26    22616
56    21328
35    14533
30    13748
5     12546
6     11918
32    11899
18    11470
0     11346
31     9252
8      8401
46     8309
14     7823
10     7729
38     7622
19     6838
13     6007
42     5992
9      5686
40     5529
58     5061
43     4840
7      4799
16     4762
51     4629
34     4615
74     4326
47     3921
59     3902
4      3726
28     3546
29     3486
44     3392
57     3023
20     2993
3      2927
49     2918
25     2917
24     2864
21     2694
36     2683
64     2659
69     2628
79     2624
39     2557
70     2409
54     2322
72     2287
33     2181
1      2096
48     2040
75     2024
66     1956
23     1919
65     1870
17     1799
63     1794
15     1754
37     1716
22     1686
27     1561
60     1547
61     1498
80     1426
77     1407
53     1397
55     1395
71     1366
52     1345
67     1287
76     1157
41     1073
68      985
11      985
78      960
45      850
62      768
73      510
"""
