from unicodedata import normalize
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def draw(label_set: list, predict_set: list, tag_set: list, n_class: int, ylabel: str="TPR", xlabel: str="FPR"):
    assert len(label_set) == len(predict_set) == len(tag_set)
    plt_figure = plt.figure()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    color_set = ["orangered", "cyan", "fuchsia", "darkorange", "mediumseagreen", "cornflowerblue", "darkorchid"]
    for plt_index in range(len(label_set)):
        assert len(label_set[plt_index]) == len(predict_set[plt_index])
        fpr_list = []
        tpr_list = []
        for index in range(n_class):
            one_hot_set = label_binarize(label_set[plt_index], classes=[0, 1, 2])
            # normalized_predict_set = (predict_set[plt_index][:, index] - np.min(predict_set[plt_index][:, index])) / (np.max(predict_set[plt_index][:, index]) - np.min(predict_set[plt_index][:, index]))
            normalized_predict_set = predict_set[plt_index][:, index]
            fpr, tpr, threshold = roc_curve(one_hot_set[:, index], normalized_predict_set)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            for i in range(len(tpr)):
                if i == 0:
                    ks_max = tpr[i] - fpr[i]
                    best_thr = threshold[i]
                elif tpr[i] - fpr[i] > ks_max:
                    ks_max = tpr[i] - fpr[i]
                    best_thr = threshold[i]
            print("{} 类别{} AUC为{} 阈值为{}".format(tag_set[plt_index], index, auc(fpr, tpr), best_thr))
        final_fpr = np.unique(np.concatenate([fpr_list[index] for index in range(n_class)]))
        final_tpr = np.zeros_like(final_fpr)
        for index in range(n_class):
            final_tpr += np.interp(final_fpr, fpr_list[index], tpr_list[index])
        final_tpr /= n_class
        final_auc = auc(final_fpr, final_tpr)
        plt.plot(final_fpr, final_tpr, color=color_set[plt_index], label="{}, AUC={:.2f}".format(tag_set[plt_index], final_auc))
    plt.legend()
    return plt_figure