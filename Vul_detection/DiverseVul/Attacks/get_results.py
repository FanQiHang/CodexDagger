import numpy as np
import os
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import roc_curve, auc


def get_result(attack_model, labels, preds, logits, path, transfer_results):

    eval_acc = np.mean(labels == preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = f1_score(labels, preds)
    fpr_, tpr_, thresholds = roc_curve(labels, logits[:, 1])
    fnr_ls = 1 - tpr_
    index = np.argmin(np.abs(fpr_ - 0.1))
    fnr_at_fpr_0_1 = fnr_ls[index]
    index = np.argmin(np.abs(fpr_ - 0.01))
    fnr_at_fpr_0_01 = fnr_ls[index]
    roc_auc = auc(fpr_, tpr_)
    # file = path + '_roc_data.npz'
    # np.savez(file, fpr_=fpr_, tpr_=tpr_)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[1, 0]).ravel()
    fpr_0 = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_0 = fn / (fn + tp) if (fn + tp) > 0 else 0
    recall_0 = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision_0 = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_0 = f1_score(labels, preds, pos_label=0)
    fpr_, tpr_, thresholds = roc_curve(labels, logits[:, 0], pos_label=0)
    fnr_ls = 1 - tpr_
    index = np.argmin(np.abs(fpr_ - 0.1))
    fnr_at_fpr_0_1_ = fnr_ls[index]
    index = np.argmin(np.abs(fpr_ - 0.01))
    fnr_at_fpr_0_01_ = fnr_ls[index]

    roc_auc_0 = auc(fpr_, tpr_)
    # file = path + '_roc_data_0.npz'
    # np.savez(file, fpr_=fpr_, tpr_=tpr_)

    transfer_results['acc'] = round(eval_acc, 4)
    transfer_results['auc'] = round(roc_auc, 4)
    transfer_results['f1-score'] = round(f1, 4)
    transfer_results['f1-score_0'] = round(f1_0, 4)
    transfer_results['fpr'] = round(fpr, 4)
    transfer_results['fpr_0'] = round(fpr_0, 4)
    transfer_results['fnr'] = round(fnr, 4)
    transfer_results['fnr_0'] = round(fnr_0, 4)
    transfer_results['precision'] = round(precision, 4)
    transfer_results['precision_0'] = round(precision_0, 4)
    transfer_results['recall'] = round(recall, 4)
    transfer_results['recall_0'] = round(recall_0, 4)
    transfer_results['fnr_at_fpr_0_1'] = round(fnr_at_fpr_0_1, 4)
    transfer_results['fnr_at_fpr_0_01'] = round(fnr_at_fpr_0_01, 4)
    transfer_results['fnr_at_fpr_0_1_0'] = round(fnr_at_fpr_0_1_, 4)
    transfer_results['fnr_at_fpr_0_01_0'] = round(fnr_at_fpr_0_01_, 4)

    return transfer_results
