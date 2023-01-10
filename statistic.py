import numpy as np
import torch
from copy import deepcopy
from typing import List

def confusion_matrix(label: torch.Tensor, predict: torch.Tensor, num_class: int) -> np.ndarray:
    cm = np.zeros((num_class, num_class))
    for i in range(num_class):
        for j in range(num_class):
            cm[i][j] = ((predict == j) & (label == i)).sum() # row is label, col is pred
    return cm.astype(np.int16)

def tp(cm: np.ndarray, target_class: int) -> int:
    return cm[target_class, target_class]

def fp(cm: np.ndarray, target_class: int) -> int:
    cm_temp = deepcopy(cm)
    cm_temp[target_class, target_class] = 0
    return np.sum(cm_temp[:, target_class])

def tn(cm: np.ndarray, target_class: int) -> int:
    cm_temp = deepcopy(cm)
    cm_temp[target_class, :] = 0
    cm_temp[:, target_class] = 0
    return np.sum(cm_temp)

def fn(cm: np.ndarray, target_class: int) -> int:
    cm_temp = deepcopy(cm)
    cm_temp[target_class, target_class] = 0
    return np.sum(cm_temp[target_class, :])

def tpr(cm: np.ndarray, target_class: int) -> list:
    _tp = tp(cm, target_class)
    _fn = fn(cm, target_class)
    denominator = (_tp + _fn)
    if denominator != 0:
        return "{}".format(_tp / denominator), "{}/{}".format(_tp, denominator)
    else:
        return "0", "0/0"

def fpr(cm: np.ndarray, target_class: int) -> list:
    _tn = tn(cm, target_class)
    _fp = fp(cm, target_class)
    denominator = (_fp + _tn)
    if denominator != 0:
        return "{}".format(_fp / denominator), "{}/{}".format(_fp, denominator)
    else:
        return "0", "0/0"

# def tnr(cm: np.ndarray, target_class: int) -> float:
#     return 1 - fpr(cm, target_class)

# def fnr(cm: np.ndarray, target_class: int) -> float:
#     return -1.0

def acc(cm: np.ndarray, target_class: int) -> list:
    _tp = tp(cm, target_class)
    _fn = fn(cm, target_class)
    _tn = tn(cm, target_class)
    _fp = fp(cm, target_class)
    denominator = (_tp + _fn + _tn + _fp)
    if denominator != 0:
        return "{}".format((_tp + _tn) / denominator), "{}/{}".format(_tp + _tn, denominator)
    else:
        return "0", "0/0"

def recall(cm: np.ndarray, target_class: int) -> list:
    _tp = tp(cm, target_class)
    _fn = fn(cm, target_class)
    denominator = (_tp + _fn)
    if denominator != 0:
        return "{}".format(_tp / denominator), "{}/{}".format(_tp, denominator)
    else:
        return "0", "0/0"

def specificity(cm:np.ndarray,target_class:int) -> list:#specificity = tn/(tn+fp)
    _tn = tn(cm, target_class)
    _fp = fp(cm, target_class)
    denominator = (_tn + _fp)
    if denominator != 0:
        return "{}".format(_tn / denominator), "{}/{}".format(_tn, denominator)
    else:
        return "0", "0/0"

def micro_f1(cm:np.ndarray) -> float:
    # micro-p = avg(TP)/(avg(TP)+avg(FP))
    # micro-r = avg(TP)/(avg(TP)+avg(FN))
    # micro-f1 = 2xmicropxmicror/(microp+micror)
    tpList = [tp(cm,i) for i in range(len(cm))]
    fpList = [fp(cm,i) for i in range(len(cm))]
    fnList = [fn(cm,i) for i in range(len(cm))]
    microP = np.mean(tpList)/(np.mean(tpList)+np.mean(fpList))
    microR = np.mean(tpList)/(np.mean(tpList)+np.mean(fnList))
    micro_F1 = 2*microP*microR/(microP+microR)
    return micro_F1


def precision(cm: np.ndarray, target_class: int) -> list:
    _tp = tp(cm, target_class)
    _fp = fp(cm, target_class)
    denominator = (_tp + _fp)
    if denominator != 0:
        return "{}".format(_tp / denominator), "{}/{}".format(_tp, denominator)
    else:
        return "0", "0/0"

def npv(cm: np.ndarray, target_class: int) -> list:
    _tn = tn(cm, target_class)
    _fn = fn(cm, target_class)
    denominator = (_tn + _fn)
    if denominator != 0:
        return "{}".format(_tn / denominator), "{}/{}".format(_tn, denominator)
    else:
        return "0", "0/0"