import numpy as np
from sklearn.metrics import roc_auc_score

def max_confidence(probs):
    return probs.max(axis=1)

def mia_auc(member_conf, nonmember_conf):
    y_true = np.concatenate([np.ones_like(member_conf), np.zeros_like(nonmember_conf)])
    y_score = np.concatenate([member_conf, nonmember_conf])
    return roc_auc_score(y_true, y_score)

def mia_from_probs(train_probs, test_probs):
    """Simple threshold attack using max softmax confidence."""
    m = max_confidence(train_probs)
    nm = max_confidence(test_probs)
    return mia_auc(m, nm)
