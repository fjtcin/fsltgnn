# Few-Shot Learning on TGNNs

> This project is based upon [DyGLib](https://github.com/yule-BUAA/DyGLib).

To solve the few-shot learning problem on temporal graph neural networks (TGNNs), we leverage the pretraining-prompt and pretraining-finetuning techniques originally used to solve few-shot learning on normal GNNs.

This project involves both edge classification and node classification tasks.

## Model

We use a TGNN model, either `TGAT`, `CAWN`, `TCL`, `GraphMixer` or `DyGFormer`, as the dynamic backbone. It does the job of integrating temporal information (timestamps) and graph structure information, along with feature vectors of edges and/or nodes, into node embeddings.

We design our predictor model, which uses these node embeddings for link prediction, edge classification, and node classification tasks.

The link prediction is a pretraining task and we do not report its ROC_AUC score. (In fact, we do not even need to evaluate its ROC_AUC score if we use loss for early stopping. There is no evaluation phase.) The predictor is similarity-based and prompt-based. Every training batch is composed of two equally sized groups of directed edges. The two groups also have exactly the same source nodes of edges. In the positive group, the destination nodes are connected to each corresponding source node, while in the negative group, they are not. The node embeddings are first multiplied by a generic task-related prompt, and the similarity of each <source node, destination node> pair is calculated and compared to a binary value of whether there exists such a directed edge between the two nodes. The similarity can be implemented as cosine-similarity, but we use MLP-similarity to improve model learning capability. The parameters of the MLP-similarity will be frozen in classification tasks if pretraining-prompt architecture is employed. If pretraining-finetuning architecture is employed, we do not use the predictor trained in this section; i.e., we only use the backbone model.

Edge classification predictor has three options: the `learnable` classifier, the `mean` classifier, and the `mlp` classifier. The `mlp` classifier uses pretraining-finetuning architecture, and the class predicted is MLP<source node embedding, destination node embedding>. The `learnable` and `mean` classifiers use pretraining-prompt architecture: When determining the class for an edge, we compare it to each prototypical edge that is considered to be the center edge for each class. By "compare," we mean the similarity is calculated. (Again, cosine-similarity is more intuitive, but similarity calculated by MLP has superior performance.) The edge is classified into the class whose prototypical edge is most similar to the edge itself. In the `learnable` classifier, the prototypical edges are learnable parameter tensors randomly initialized. In the `mean` setting, the prototypical edges are the mean values of all current node embeddings of the same classes. The `mean` prototypical edges need to be calculated over all training data for every epoch (since the node embedding changes during training), so training time becomes $\Omega(n^2)$ from $\Omega(n)$, where $n$ is the number of training data. However, it is still computable since we are dealing with a few-shot learning problem.

The node classification task is converted to edge classification. We add a self-loop for the node to be classified and classify the self-loop instead.

## Experiments

### Setup Environment

```bash
conda create -n fsltgnn python=3.10
conda activate fsltgnn
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pandas tqdm scikit-learn
```

### Prepare Dataset

For edge classification:

| dataset | num of nodes | num of edges | edge label distribution | node features | edge features |
| -------- | ------- | ------- | ------- | ------- | ------- |
| hyperlink | 35776 | 286561 | 21070(0), 265491(1) | 300* | 86 |
| mooc      | 7144 | 411749 | 407683(0), 4066(1) | none | 4 |
| reddit    | 10984 | 672447 | 672081(0), 366(1) | none | 172 |
| wikipedia | 9227 | 157474 | 157257(0), 217(1) | none | 172 |

[*] 14102 out of 35776 nodes do not have node features. We use 300 zeros to fill.

For node classification:

| dataset | num of nodes | num of edges | node label distribution | node features | edge features |
| -------- | ------- | ------- | ------- | ------- | ------- |
| gdelt | 3049 | 1855541 | 41436(0), 28268(1), 22616(2), 21328(3) | 413 | 182 |

The sum of nodes in "node label distribution" is larger than "num of nodes". This is because a node has a label for each timestamp, and there are multiple timestamps for a node. The node classification problem is to predict the label of a given node at a given time.

#### hyperlink

> We integrate two reddit embedding datasets [[1](https://snap.stanford.edu/data/soc-RedditHyperlinks.html), [2](https://snap.stanford.edu/data/web-RedditEmbeddings.html)] into the `hyperlink` dataset. The hyperlink network represents the directed connections between two subreddits.

Download [soc-redditHyperlinks-body.tsv](https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv) and [web-redditEmbeddings-subreddits.csv](https://snap.stanford.edu/data/web-redditEmbeddings-subreddits.csv) to `DG_data/hyperlink/`.

Run `python preprocess_data/preprocess_hyperlink.py`.

#### mooc

> The mooc dataset is from [DGB](https://github.com/fpour/DGB). It is a student interaction network formed from online course content units such as problem sets and videos. See their [paper](https://arxiv.org/pdf/2207.10128) for more details.

Download the [mooc](https://zenodo.org/records/7213796/files/mooc.zip) dataset and unzip it to `./DG_data/`.

Run `python preprocess_data/preprocess_data.py --dataset_name mooc`.

#### reddit

> The reddit dataset is from [DGB](https://github.com/fpour/DGB). It models subreddits' posted spanning one month, where the nodes are users or posts and the edges are the timestamped posting requests. See their [paper](https://arxiv.org/pdf/2207.10128) for more details.

Download the [reddit](https://zenodo.org/records/7213796/files/reddit.zip) dataset and unzip it to `./DG_data/`.

Run `python preprocess_data/preprocess_data.py --dataset_name reddit`.

#### wikipedia

> The wikipedia dataset is from [DGB](https://github.com/fpour/DGB). It consists of edits on Wikipedia pages over one month. Editors and wiki pages are modelled as nodes, and the timestamped posting requests are edges. See their [paper](https://arxiv.org/pdf/2207.10128) for more details.

Download the [wikipedia](https://zenodo.org/records/7213796/files/wikipedia.zip) dataset and unzip it to `./DG_data/`.

Run `python preprocess_data/preprocess_data.py --dataset_name wikipedia`.

#### gdelt

> The gdelt dataset is from [tgl](https://github.com/amazon-science/tgl). It is a Temporal Knowledge Graph originated from the Event Database in GDELT 2.0 which records events happening in the world from news and articles in over 100 languages every 15 minutes. See their [paper](https://arxiv.org/pdf/2203.14883) for more details.

Download the original gdelt dataset (42 GiB, 4 files).

```bash
wget -P ./DG_data/gdelt/ https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/node_features.pt
wget -P ./DG_data/gdelt/ https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/labels.csv
wget -P ./DG_data/gdelt/ https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/edges.csv
wget -P ./DG_data/gdelt/ https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/edge_features.pt
```

Run `python preprocess_data/preprocess_gdelt.py`. (at least 40 GB RAM is needed)

> We minimized the original large dataset with 100:1 edge sampling like in this [GraphMixer paper](https://arxiv.org/pdf/2302.11636).

### Run

pretraining-prompt architecture:

```bash
python link_prediction.py --dataset_name hyperlink --model_name GraphMixer --load_best_configs --num_runs 3
python classification.py --classifier learnable --dataset_name hyperlink --model_name GraphMixer --load_best_configs --num_runs 3
python classification.py --classifier mean --dataset_name hyperlink --model_name GraphMixer --load_best_configs --num_runs 3
```

pretraining-finetuning architecture:

```bash
python link_prediction.py --dataset_name hyperlink --model_name GraphMixer --load_best_configs --num_runs 3
python classification.py --classifier mlp --dataset_name hyperlink --model_name GraphMixer --load_best_configs --num_runs 3
```

end-to-end training (baseline):

```bash
python classification_e2e.py --dataset_name hyperlink --model_name GraphMixer --load_best_configs --num_runs 3
```

> `dataset_name` can be any dataset name from the [Prepare Dataset section](#prepare-dataset).
> Available backbone models: `TGAT`, `CAWN`, `TCL`, `GraphMixer` and `DyGFormer`, see [DyGLib](https://github.com/yule-BUAA/DyGLib) for more model information.

### Results

All Edge classification datasets we have found are for binary classification. A ROC_AUC score from 3 runs is repoted.

We have only found one dataset that is availble for node classification. We preprocessed the `gdelt` dataset for 4-class classification. An Accuracy score from 1 run is reported.

The "unseen" tag means an edge/node that is in training data will not appear in the test data, although timestamps are different. In other words, it is the test data with repeating edge/nodes from the training data removed.

We trained the temporal dataset for 5% of the timestamps.

#### GraphMixer backbone

| dataset | learnable classifier | mean classifier | MLP fine-tuning | end-to-end baseline |
| -------- | ------- | ------- | ------- | ------- |
| hyperlink          | 0.7298 ± 0.0071 | 0.7177 ± 0.0009 | 0.7218 ± 0.0019 | 0.6998 ± 0.0031 |
| hyperlink (unseen) | 0.7101 ± 0.0085 | 0.7030 ± 0.0013 | 0.6932 ± 0.0025 | 0.6555 ± 0.0064 |
| mooc               | 0.5988 ± 0.0447 | 0.6091 ± 0.0226 | 0.7486 ± 0.0063 | 0.7028 ± 0.0023 |
| mooc (unseen)      | 0.6046 ± 0.0416 | 0.6147 ± 0.0235 | 0.7550 ± 0.0039 | 0.7011 ± 0.0018 |
| wikipedia          | 0.5525 ± 0.1233 | 0.7161 ± 0.0152 | 0.5441 ± 0.2306 | 0.8334 ± 0.0129 |
| wikipedia (unseen) | 0.5562 ± 0.1263 | 0.7219 ± 0.0164 | 0.5362 ± 0.2303 | 0.8028 ± 0.0203 |

| dataset | learnable classifier | mean classifier | MLP fine-tuning | end-to-end baseline |
| -------- | ------- | ------- | ------- | ------- |
| gdelt              | 0.3788          | 0.3853          | 0.4596          | 0.4182          |
| gdelt (unseen)     | 0.3717          | 0.3676          | 0.3567          | 0.3137          |

## Conclusion

We achieved 9.0% and 7.8% improvement on `hyperlink` for pretraining-prompt (learnable classifier) and pretraining-finetuning respectively, campared to the end-to-end baseline.

## Appendix

There are two standalone python scripts in the `utils/` directory.

### `check_mooc.py`: different versions of the MOOC dataset

### `one_hot_speed_test.py`: performance measurement of one-hot encoding methods

example output:

```text
Scatter Time: 0.015410900115966797
Advanced Indexing Time: 0.10599017143249512
PyTorch Time: 0.02737879753112793
```
