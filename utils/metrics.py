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


def get_edge_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor, fp='/dev/null'):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, num_classes)
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.numpy(force=True)
    labels = labels.numpy(force=True)

    report = classification_report(y_true=labels, y_pred=predicts>0.5, output_dict=True)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)
    average_precision = average_precision_score(y_true=labels, y_score=predicts)

    with open(fp, 'a') as f:
        json.dump(report | {'roc_auc': roc_auc, 'average_precision': average_precision}, f, indent=4)
    return {'roc_auc': roc_auc}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor, fp='/dev/null'):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, num_classes)
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.argmax(axis=1).numpy(force=True)
    labels = labels.numpy(force=True)

    report = classification_report(y_true=labels, y_pred=predicts, output_dict=True)

    with open(fp, 'a') as f:
        json.dump(report, f, indent=4)
    return {'accuracy': report['accuracy']}
