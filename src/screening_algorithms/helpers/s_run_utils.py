import numpy as np
from scipy.special import binom
from fusion_algorithms.algorithms_utils import input_adapter
from fusion_algorithms.em import expectation_maximization


def assign_filters(items, filters_num, values_count, filters_select_est, filters_acc_est, prior_prob_pos=None):
    stop_score = 15
    filters_assigned = []
    items_new = []
    filters = range(filters_num)
    for item_index in items:
        classify_score = []
        n_min_list = []
        joint_prob_votes_neg = [1.] * filters_num
        for filter_index in filters:
            filter_acc = filters_acc_est[filter_index]
            filter_select = filters_select_est[filter_index]
            if prior_prob_pos:
                prob_item_neg = 1 - prior_prob_pos[item_index * filters_num + filter_index]
            else:
                prob_item_neg = filters_select_est[filter_index]
            pos_c, neg_c = values_count[item_index*filters_num + filter_index]
            for n in range(1, 11):
                # new value is negative
                prob_vote_neg = filter_acc*prob_item_neg + (1 - filter_acc)*(1 - prob_item_neg)
                joint_prob_votes_neg[filter_index] *= prob_vote_neg
                term_neg = binom(pos_c+neg_c+n, neg_c+n) * filter_acc**(neg_c+n)\
                           * (1-filter_acc)**pos_c * filter_select
                term_pos = binom(pos_c+neg_c+n, pos_c) * filter_acc**pos_c \
                             * (1-filter_acc)**(neg_c+n) * (1-filter_select)
                prob_item_pos = term_pos * prob_vote_neg / (term_neg + term_pos)
                prob_item_neg = 1 - prob_item_pos
                if prob_item_neg >= 0.99:
                    classify_score.append(joint_prob_votes_neg[filter_index]/n)
                    n_min_list.append(n)
                    break
                elif n == 10:
                    classify_score.append(joint_prob_votes_neg[filter_index]/n)
                    n_min_list.append(n)

        filter_ = classify_score.index(max(classify_score))
        n_min = n_min_list[filter_]
        joint_prob = joint_prob_votes_neg[filter_]

        if n_min / joint_prob < stop_score:
            filters_assigned.append(filter_)
            items_new.append(item_index)

    return filters_assigned, items_new


def classify_items_baseround(items, filters_num, values_prob):
    items_classified = {}
    items_to_classify = []
    trsh = 0.99
    for item_index in items:
        prob_pos = 1.
        for filter_index in range(filters_num):
            prob_pos *= values_prob[item_index*filters_num + filter_index][0]
        prob_neg = 1 - prob_pos

        if prob_neg > trsh:
            items_classified[item_index] = 0
        elif prob_pos > trsh:
            items_classified[item_index] = 1
        else:
            items_to_classify.append(item_index)

    return items_classified, items_to_classify


def classify_items(items, filters_num, values_count, thrs, filters_acc_est, filters_select_est):
    items_classified = {}
    items_to_classify = []

    for item_index in items:
        prob_item_pos = 1.
        for filter_index in range(filters_num):
            filter_acc = filters_acc_est[filter_index]
            filter_select = filters_select_est[filter_index]

            pos_c, neg_c = values_count[item_index*filters_num + filter_index]
            if pos_c == 0 and neg_c == 0:
                prob_filter_pos = 1 - filter_select
            else:
                term_pos = binom(pos_c+neg_c, pos_c) * filter_acc**pos_c * (1-filter_acc)**neg_c * (1-filter_select)
                term_neg = binom(pos_c+neg_c, neg_c) * filter_acc**neg_c * (1-filter_acc)**pos_c * filter_select
                prob_filter_pos = term_pos / (term_pos + term_neg)
            prob_item_pos *= prob_filter_pos
        prob_item_neg = 1 - prob_item_pos

        if prob_item_neg > thrs:
            items_classified[item_index] = 0
        elif prob_item_pos > thrs:
            items_classified[item_index] = 1
        else:
            items_to_classify.append(item_index)

    return items_classified, items_to_classify


def generate_votes(ground_truth, items, filters_num, items_per_worker, workers_accuracy, filters_dif, filters_assigned):
    votes = []
    items_num = len(items)
    workers_num = 1 if items_num < items_per_worker else items_num // items_per_worker
    for worker_index in range(workers_num):
        # get worker's accuracy
        worker_acc_pos = workers_accuracy[1].pop()
        workers_accuracy[1].insert(0, worker_acc_pos)
        worker_acc_neg = workers_accuracy[0].pop()
        workers_accuracy[0].insert(0, worker_acc_neg)

        filter_item_pair = zip(filters_assigned[worker_index*items_per_worker:
                               worker_index*items_per_worker + items_per_worker],
                               items[worker_index*items_per_worker:worker_index*items_per_worker + items_per_worker])
        for filter_index, item_index in filter_item_pair:
            # update the worker's accuracy on the current item
            is_item_pos = sum(ground_truth[item_index*filters_num:item_index*filters_num + filters_num]) == 0
            if is_item_pos:
                worker_acc = worker_acc_pos
            else:
                worker_acc = worker_acc_neg

            # generate vote
            value_gt = ground_truth[item_index*filters_num + filter_index]
            cr_dif = filters_dif[filter_index]
            if np.random.binomial(1, worker_acc * cr_dif if worker_acc * cr_dif <= 1. else 1.):
                vote = value_gt
            else:
                vote = 1 - value_gt
            votes.append(vote)

    return votes


def update_votes_count(values_count, filters_num, filters_assigned, votes, items):
    for filter_index, vote, item_index in zip(filters_assigned, votes, items):
        if vote:
            values_count[item_index*filters_num + filter_index][1] += 1
        else:
            values_count[item_index * filters_num + filter_index][0] += 1


def update_filters_select(items_num, filters_num, filters_acc_est, filters_select_est, values_count):
    apply_filters_prob = [[] for _ in range(filters_num)]
    for item_index in range(items_num):
        for filter_index in range(filters_num):
            filter_acc = filters_acc_est[filter_index]
            filter_select = filters_select_est[filter_index]
            pos_c, neg_c = values_count[item_index*filters_num + filter_index]

            term_pos = binom(pos_c+neg_c, pos_c) * filter_acc**pos_c * (1-filter_acc)**neg_c * (1-filter_select)
            term_neg = binom(pos_c+neg_c, neg_c) * filter_acc**neg_c * (1-filter_acc)**pos_c * filter_select
            prob_filter_neg = term_neg / (term_pos + term_neg)
            apply_filters_prob[filter_index].append(prob_filter_neg)

    filters_select_new = [np.mean(i) for i in apply_filters_prob]
    return filters_select_new

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
