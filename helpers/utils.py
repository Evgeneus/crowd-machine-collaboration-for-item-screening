from fusion_algorithms.algorithms_utils import input_adapter
from fusion_algorithms.em import expectation_maximization

import numpy as np


# simulate workers that pass a set of test questions
def simulate_workers(worker_tests, cheaters_prop):
    acc_passed_distr = [[], []]
    for _ in range(100000):
        result = simulate_quiz(worker_tests, cheaters_prop)
        if not isinstance(result, str):
            acc_passed_distr[0].append(result[0])
            acc_passed_distr[1].append(result[1])
    return acc_passed_distr


def simulate_quiz(worker_tests, cheaters_prop):
    # decide if a worker a cheater
    if np.random.binomial(1, cheaters_prop):
        worker_type = 'rand_ch'
        worker_acc_neg, worker_acc_pos = 0.5, 0.5
    else:
        worker_type = 'worker'
        worker_acc_pos = 0.5 + (np.random.beta(1, 1) * 0.5)
        worker_acc_neg = worker_acc_pos + 0.1 if worker_acc_pos + 0.1 <= 1. else 1.

    for item_index in range(worker_tests):
        # decide if the test item is positive or negative (50+/50-)
        if np.random.binomial(1, 0.5):
            # if worker is mistaken exclude him
            if not np.random.binomial(1, worker_acc_pos):
                return worker_type
        else:
            if not np.random.binomial(1, worker_acc_neg):
                return worker_type
    return worker_acc_neg, worker_acc_pos, worker_type


def compute_metrics(items, gt, lr, filters_num):
    # obtain ground_truth scope values for items
    gt_scope = []
    for item_index in range(len(items)):
        if sum([gt[item_index*filters_num + filter_index] for filter_index in range(filters_num)]):
            gt_scope.append(0)
        else:
            gt_scope.append(1)
    # Positive == Inclusion (Relevant)
    # Negative == Exclusion (Not relevant)
    fp = 0.
    fn = 0.
    tp = 0.
    tn = 0.
    for cl_val, gt_val in zip(items, gt_scope):
        if gt_val and not cl_val:
            fn += 1
        if not gt_val and cl_val:
            fp += 1
        if gt_val and cl_val:
            tp += 1
        if not gt_val and not cl_val:
            tn += 1
    recall = 100 * tp / (tp + fn)
    precision = 100 * tp / (tp + fp)
    loss = (fn * lr + fp) / len(items)
    beta = 1. / lr
    f_beta = (beta + 1) * precision * recall / (beta * recall + precision)
    return loss, recall, precision, f_beta, fp


def estimate_filters_property(votes, filters_num, items_num, items_per_worker, votes_per_item):
    psi = input_adapter(votes)
    n = (items_num // items_per_worker) * votes_per_item
    filters_select_est = []
    filters_acc_est = []
    for filter_index in range(filters_num):
        item_filter_votes = psi[filter_index::filters_num]
        filter_acc_list, filter_select_list = expectation_maximization(n, items_num, item_filter_votes)
        filter_acc = np.mean(filter_acc_list)
        filter_select = 0.
        for i in filter_select_list:
            i_prob = [0., 0.]
            for i_index, i_p in i.items():
                i_prob[i_index] = i_p
            filter_select += i_prob[1]
        filter_select /= items_num
        filters_select_est.append(filter_select)
        filters_acc_est.append(filter_acc)

    return filters_select_est, filters_acc_est
