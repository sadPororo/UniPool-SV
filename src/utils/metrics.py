""" Metric functions for the evalaution of the benchmark systems

[Metrics]
For Speaker Identification (Multi-calss Classification)
- Accuracy (ACC)
- F1-macro score (F1)

For Speaker Verification (Binary-calss Classification)
- Equal Error Rate (EER)
- Minimum Detection Cost Function (minDCF)
  * The implemetation of "EER()" and "minDCF()" from the "SpeechBrain" is adopted, 
    and I wrap the code up into "get_EER()" and "get_minDCF()" for the intuitive usage.

[Code References]
    EER: https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/utils/metric_stats.html#EER
    minDCF: https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/utils/metric_stats.html#minDCF
"""
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


def get_Accuracy(y_target:torch.Tensor, y_pred:torch.Tensor):
    """ Get accuracy of the model prediction

    Args:
        y_target (torch.Tensor): 1D-array with class label indice in {0, C-1}, where C is the number of the classes
        y_pred (torch.Tensor): The prediction of the model, 1D-array with class label indice in {0, C-1}.

    Returns:
        the accuracy of given evaluation
    """
    assert len(y_target) == len(y_pred)
    return (y_target == y_pred).sum() / len(y_target)


def get_F1score(y_target:torch.Tensor, y_pred:torch.Tensor):
    """ Get F1-score of the model prediction

    Args:
        y_target (torch.Tensor): 1D-array with class label indice in {0, C-1}, where C is the number of the classes
        y_pred (torch.Tensor): The prediction of the model, 1D-array with class label indice in {0, C-1}.

    Returns:
        the F1 (macro) score of given evaluation
    """
    return f1_score(y_target.numpy(), y_pred.numpy(), average='macro')


#%%

def get_EER(y_target:torch.Tensor, y_pred:torch.Tensor, pos_label=1):
    """ Wrapper function of EER calculation, returns EER score & corresponding threshold value.

    Args:
        y_target (torch.Tensor): 1D-array with binary label in {0, 1}.
        y_pred (torch.Tensor): 1D-array of similarity prediction between two audio samples (range in [0, 1]).
        pos_label (int, optional): definition of the positive. Defaults to 1.
        
    Variables:
        pred_for_tp: Predictions for Positive-labelled samples
        pred_for_tn: Predictions for Negative-labelled samples
    """
    pred_for_tp = y_pred[(y_target==pos_label)]
    pred_for_tn = y_pred[(y_target!=pos_label)]
    
    assert len(pred_for_tp) + len(pred_for_tn) == len(y_target)
    assert len(pred_for_tp) + len(pred_for_tn) == len(y_pred)
    
    eer, threshold = EER(pred_for_tp, pred_for_tn)

    return eer, threshold


def get_minDCF(y_target:torch.Tensor, y_pred:torch.Tensor, c_miss=1.0, c_fa=1.0, p_target=0.01, pos_label=1):
    """ Wrapper function of minDCF calculation, returns minDCF score & corresponding threshold value.

        ** False Positive 
                False Accept (FA)       : target=False vs. prediction=True
           False Negative 
                False Reject (FR; Miss) : target=True vs. prediction = False

    Args:
        y_pred (np.1Darray | list): 1D-array of similarity prediction between two audio samples (range in [0, 1]).
        c_miss (float, optional): A cost for False-negative error. Defaults to 1.
        c_fa (float, optional): A cost for False-positive error. Defaults to 1.
        p_target (float, optional): prior probability of positive sample occurence. Defaults to 0.01.
        pos_label (int, optional): definition of the positive. Defaults to 1.        
    """
    pred_for_tp = y_pred[(y_target==pos_label)]
    pred_for_tn = y_pred[(y_target!=pos_label)]
    
    assert len(pred_for_tp) + len(pred_for_tn) == len(y_target)
    assert len(pred_for_tp) + len(pred_for_tn) == len(y_pred)

    minCost, threshold = minDCF(pred_for_tp, pred_for_tn, c_miss, c_fa, p_target)
    
    return minCost, threshold


def get_FAR(y_target:torch.Tensor, y_pred:torch.Tensor, threshold=None):
    """ Get 'False Acceptance Rate' (FAR | FPR) of the model prediction
    As defined by, FAR = FP / (FP + TN)

    Args:
        y_target (torch.Tensor): 1D-array with class label indice in {0, C-1}, where C is the number of the classes
        y_pred (torch.Tensor): The prediction of the model, 1D-array with class label indice in {0, C-1}.
        threshold (None | float): measure FAR with the threshold value if given, else will get the list of FAR using the default thresholds

    Returns:
        the FAR score of given evaluation
    """
    assert len(y_target) == len(y_pred)
    
    y_target = y_target.to(bool)
    
    if threshold is None:
        threshold = torch.arange(0, 1, 0.01)[1:] # 0.01 ~ 0.99
    else:
        threshold = torch.Tensor([threshold])
        
    far = []
    for t in threshold:
        pred_for_far = y_pred > t
        
        fp = (~y_target[pred_for_far]).sum()        
        far.append(fp / (~y_target).sum())
        
    return torch.Tensor(far), threshold
    
    
def get_FRR(y_target:torch.Tensor, y_pred:torch.Tensor, threshold=None):
    """ Get 'False Rejection Rate' (FRR | FNR) of the model prediction
    As defined by, FRR = FN / (TP + TN)

    Args:
        y_target (torch.Tensor): 1D-array with class label indice in {0, C-1}, where C is the number of the classes
        y_pred (torch.Tensor): The prediction of the model, 1D-array with class label indice in {0, C-1}.
        threshold (None | float): measure FAR with the threshold value if given, else will get the list of FAR using the default thresholds

    Returns:
        the FAR score of given evaluation
    """
    assert len(y_target) == len(y_pred)
    
    y_target = y_target.to(bool)
    
    if threshold is None:
        threshold = torch.arange(0, 1, 0.01)[1:] # 0.01 ~ 0.99
    else:
        threshold = torch.Tensor([threshold])
        
    frr = []
    for t in threshold:
        pred_for_frr = y_pred > t
        
        fn = (y_target[~pred_for_frr]).sum()
        frr.append(fn / (y_target).sum())
        
    return torch.Tensor(frr), threshold


def get_DETcurve(y_target:torch.Tensor, y_pred:torch.Tensor, threshold=None, figures=None):
    """ Visualize 'DET (Detection Error Tradeoff) curve' 
    
    Args:
        y_target (torch.Tensor): 1D-array with class label indice in {0, C-1}, where C is the number of the classes
        y_pred (torch.Tensor): The prediction of the model, 1D-array with class label indice in {0, C-1}.
        threshold (None | float): measure FAR with the threshold value if given, else will get the list of FAR using the default thresholds

    Returns:
        the DET curve of given evaluation
    """
    threshold
    fars, threshold = get_FAR(y_target, y_pred)
    frrs, _         = get_FRR(y_target, y_pred)
    
    if figures is not None:
        fig, ax = figures
    else:
        fig, ax = plt.subplots()
    ax.plot(fars * 100, frrs * 100)
    ax.set_yscale('log'); ax.set_xscale('log')    
    
    ticks_to_use = [0.1,0.2,0.5,
                    1,2,5,
                    10,20,50,80]

    ax.set_xlabel('False Acceptance Rate (FAR; %)')
    ax.set_ylabel('False Rejection Rate (FRR; %)')
    ax.set_xticks(ticks_to_use);      ax.set_yticks(ticks_to_use)
    ax.set_xticklabels(ticks_to_use); ax.set_yticklabels(ticks_to_use)

    ax.axis([0.1,95,0.1,95])    
    ax.grid(True, linestyle='--')

    return fig, ax    
    
    
#%%

# Code Reference: https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/utils/metric_stats.html#EER
def EER(positive_scores, negative_scores):
    """Computes the EER (and its threshold).

    Arguments
    ---------
    positive_scores : torch.tensor
        The scores from entries of the same class.
    negative_scores : torch.tensor
        The scores from entries of different classes.

    Example
    -------
    >>> positive_scores = torch.tensor([0.6, 0.7, 0.8, 0.5])
    >>> negative_scores = torch.tensor([0.4, 0.3, 0.2, 0.1])
    >>> val_eer, threshold = EER(positive_scores, negative_scores)
    >>> val_eer
    0.0
    """

    # Computing candidate thresholds
    thresholds, _ = torch.sort(torch.cat([positive_scores, negative_scores]))
    thresholds = torch.unique(thresholds)

    # Adding intermediate thresholds
    interm_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, interm_thresholds]))

    # Variable to store the min FRR, min FAR and their corresponding index
    min_index = 0
    final_FRR = 0
    final_FAR = 0

    for i, cur_thresh in enumerate(thresholds):
        pos_scores_threshold = positive_scores <= cur_thresh
        FRR = (pos_scores_threshold.sum(0)).float() / positive_scores.shape[0]
        del pos_scores_threshold

        neg_scores_threshold = negative_scores > cur_thresh
        FAR = (neg_scores_threshold.sum(0)).float() / negative_scores.shape[0]
        del neg_scores_threshold

        # Finding the threshold for EER
        if (FAR - FRR).abs().item() < abs(final_FAR - final_FRR) or i == 0:
            min_index = i
            final_FRR = FRR.item()
            final_FAR = FAR.item()

    # It is possible that eer != fpr != fnr. We return (FAR  + FRR) / 2 as EER.
    EER = (final_FAR + final_FRR) / 2

    return float(EER), float(thresholds[min_index])


# Code Reference: https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/utils/metric_stats.html#minDCF
def minDCF(positive_scores, negative_scores, c_miss=1.0, c_fa=1.0, p_target=0.01):
    """Computes the minDCF metric normally used to evaluate speaker verification
    systems. The min_DCF is the minimum of the following C_det function computed
    within the defined threshold range:

    C_det =  c_miss * p_miss * p_target + c_fa * p_fa * (1 -p_target)

    where p_miss is the missing probability and p_fa is the probability of having
    a false alarm.

    Arguments
    ---------
    positive_scores : torch.tensor
        The scores from entries of the same class.
    negative_scores : torch.tensor
        The scores from entries of different classes.
    c_miss : float
         Cost assigned to a missing error (default 1.0).
    c_fa : float
        Cost assigned to a false alarm (default 1.0).
    p_target: float
        Prior probability of having a target (default 0.01).


    Example
    -------
    >>> positive_scores = torch.tensor([0.6, 0.7, 0.8, 0.5])
    >>> negative_scores = torch.tensor([0.4, 0.3, 0.2, 0.1])
    >>> val_minDCF, threshold = minDCF(positive_scores, negative_scores)
    >>> val_minDCF
    0.0
    """

    # Computing candidate thresholds
    thresholds, _ = torch.sort(torch.cat([positive_scores, negative_scores]))
    thresholds = torch.unique(thresholds)

    # Adding intermediate thresholds
    interm_thresholds = (thresholds[0:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, interm_thresholds]))

    # Computing False Rejection Rate (miss detection)
    positive_scores = torch.cat(
        len(thresholds) * [positive_scores.unsqueeze(0)]
    )
    pos_scores_threshold = positive_scores.transpose(0, 1) <= thresholds
    p_miss = (pos_scores_threshold.sum(0)).float() / positive_scores.shape[1]
    del positive_scores
    del pos_scores_threshold

    # Computing False Acceptance Rate (false alarm)
    negative_scores = torch.cat(
        len(thresholds) * [negative_scores.unsqueeze(0)]
    )
    neg_scores_threshold = negative_scores.transpose(0, 1) > thresholds
    p_fa = (neg_scores_threshold.sum(0)).float() / negative_scores.shape[1]
    del negative_scores
    del neg_scores_threshold

    c_det = c_miss * p_miss * p_target + c_fa * p_fa * (1 - p_target)
    c_min, min_index = torch.min(c_det, dim=0)

    return float(c_min), float(thresholds[min_index])


