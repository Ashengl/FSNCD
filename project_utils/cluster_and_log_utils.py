from project_utils.cluster_utils import cluster_acc, np, linear_assignment
from torch.utils.tensorboard import SummaryWriter
from typing import List


def split_cluster_acc_v1(y_true, y_pred, mask, n_way=5):

    """
    Evaluate clustering metrics on two subsets of data, as defined by the mask 'mask'
    (Mask usually corresponding to `Old' and `New' classes in GCD setting)
    :param targets: All ground truth labels
    :param preds: All predictions
    :param mask: Mask defining two subsets
    :return:
    """

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    weight = mask.mean()

    old_acc = cluster_acc(y_true[n_way:][mask], y_pred[n_way:][mask])
    new_acc = cluster_acc(y_true[n_way:][~mask], y_pred[n_way:][~mask])
    total_acc = weight * old_acc + (1 - weight) * new_acc

    return total_acc, old_acc, new_acc


def split_cluster_acc_v2(y_true, y_pred, mask, n_way=5):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[n_way:][mask])
    new_classes_gt = set(y_true[n_way:][~mask])

    index = [i not in y_pred[:n_way] for i in y_pred]
    base_ind_map = {i:j for i, j in zip(y_true[:n_way], y_pred[:n_way])}

    if y_pred.size != y_true.size:
        print(1)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w_base = np.zeros((D, D), dtype=int)
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred[index].size):
        w_base[y_pred[index][i], y_true[index][i]] += 1
    for i in range(y_pred.size-n_way):
        w[y_pred[i+n_way], y_true[i+n_way]] += 1

    ind = linear_assignment(w_base.max() - w_base)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    ind_map_T = {i: j for i, j in ind}
    for key in base_ind_map.keys():
        ind_map[ind_map_T[base_ind_map[key]]] = ind_map[key]
        ind_map[key] = base_ind_map[key]
        ind_map_T = {v: k for k, v in ind_map.items()}
    y_true = np.array([ind_map[i] for i in y_true])
    total_acc = np.array(y_true[n_way:] == y_pred[n_way:]).mean()
    old_acc = np.array(y_true[n_way:][mask] == y_pred[n_way:][mask]).mean() if len(y_pred[n_way:][mask])!=0 else np.nan
    new_acc = np.array(y_true[n_way:][~mask] == y_pred[n_way:][~mask]).mean() if len(y_pred[n_way:][~mask])!=0 else np.nan

    return total_acc, old_acc, new_acc


EVAL_FUNCS = {
    'v1': split_cluster_acc_v1,
    'v2': split_cluster_acc_v2,
}

def log_accs_from_preds(y_true, y_pred, mask, eval_funcs: List[str], save_name: str, T: int=None, writer: SummaryWriter=None,
                        print_output=False, n_way=5):

    """
    Given a list of evaluation functions to use (e.g ['v1', 'v2']) evaluate and log ACC results

    :param y_true: GT labels
    :param y_pred: Predicted indices
    :param mask: Which instances belong to Old and New classes
    :param T: Epoch
    :param eval_funcs: Which evaluation functions to use
    :param save_name: What are we evaluating ACC on
    :param writer: Tensorboard logger
    :return:
    """


    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    for i, f_name in enumerate(eval_funcs):

        acc_f = EVAL_FUNCS[f_name]
        all_acc, old_acc, new_acc = acc_f(y_true, y_pred, mask, n_way=n_way)
        log_name = f'{save_name}_{f_name}'

        if writer is not None:
            writer.add_scalars(log_name,
                               {'Old': old_acc, 'New': new_acc,
                                'All': all_acc}, T)

        if i == 0:
            to_return = (all_acc, old_acc, new_acc)

        if print_output:
            print_str = f'Epoch {T}, {log_name}: All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}'
            print(print_str)

    return to_return