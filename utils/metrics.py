import numpy as np
import torch
import json
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.numpy(force=True)
    labels = labels.numpy(force=True)

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'average_precision': average_precision, 'roc_auc': roc_auc}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.numpy(force=True)
    labels = labels.numpy(force=True)

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}


def get_edge_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor, binary=True, fp='/dev/null'):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, num_classes)
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    probs = predicts.softmax(dim=-1).numpy(force=True)
    predicts = predicts.argmax(dim=-1).numpy(force=True)
    labels = labels.numpy(force=True)

    report = classification_report(y_true=labels, y_pred=predicts, output_dict=True)
    try:
        if binary:
            roc_auc = roc_auc_score(y_true=labels, y_score=probs[:, 1])
            roc_auc_weighted = roc_auc_score(y_true=labels, y_score=probs[:, 1], average='weighted')
        else:
            roc_auc = roc_auc_score(y_true=labels, y_score=probs, average='macro', multi_class='ovo')
            roc_auc_weighted = roc_auc_score(y_true=labels, y_score=probs, average='weighted', multi_class='ovo')
    except ValueError:
        roc_auc = np.nan
        roc_auc_weighted = np.nan

    with open(fp, 'a') as f:
        json.dump(report | {'roc_auc': roc_auc, 'roc_auc_weighted': roc_auc_weighted}, f, indent=4)
    return {'roc_auc_weighted': roc_auc_weighted}
