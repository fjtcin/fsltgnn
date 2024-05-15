import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

mooc_graph_df = pd.read_csv('./processed_data/mooc/ml_mooc.csv')
mooc_edge_raw_features = np.load('./processed_data/mooc/ml_mooc.npy')
mooc_node_raw_features = np.load('./processed_data/mooc/ml_mooc_node.npy')

moocact_graph_df = pd.read_csv('./processed_data/moocact/ml_moocact.csv')
moocact_edge_raw_features = np.load('./processed_data/moocact/ml_moocact.npy')
moocact_node_raw_features = np.load('./processed_data/moocact/ml_moocact_node.npy')

assert_frame_equal(mooc_graph_df, moocact_graph_df)
assert np.allclose(mooc_edge_raw_features, moocact_edge_raw_features)
assert(np.array_equal(mooc_node_raw_features, moocact_node_raw_features))
