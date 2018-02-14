import numpy as np


def generate_gold_data(items_num, filters_select):
    gold_data = []
    for item_index in range(items_num):
        for filter_select in filters_select:
            if np.random.binomial(1, filter_select):
                val = 1
            else:
                val = 0
            gold_data.append(val)
    return gold_data


def generate_votes_gt(items_num, filters_select, items_per_worker, worker_tests, workers_accuracy, filters_dif, gt=None):
    if not gt:
        gt = generate_gold_data(items_num, filters_select)
        is_gt_genereated = True
    else:
        is_gt_genereated = False
    workers_accuracy_neg, workers_accuracy_pos = workers_accuracy

    # generate votes
    # on a page a worker see items_per_worker tasks (crowdflower style)
    pages_num = items_num // items_per_worker
    filters_num = len(filters_select)
    votes = {}
    for item_filter_index in range(pages_num*items_per_worker*filters_num):
        votes[item_filter_index] = {}
    for page_index in range(pages_num):
        for i in range(worker_tests):
            worker_id = page_index * worker_tests + i
            w_acc_pos = workers_accuracy_pos.pop()
            workers_accuracy[1].insert(0, w_acc_pos)
            w_acc_neg = workers_accuracy_neg.pop()
            workers_accuracy[0].insert(0, w_acc_neg)
            for item_index in range(page_index*items_per_worker, page_index*items_per_worker + items_per_worker, 1):
                filter_item_indices = range(item_index*filters_num, item_index*filters_num + filters_num, 1)
                is_item_pos = sum([gt[i] for i in filter_item_indices]) == 0
                if is_item_pos:
                    worker_acc = w_acc_pos
                else:
                    worker_acc = w_acc_neg
                for item_filter_index, f_diff in zip(filter_item_indices, filters_dif):
                    if np.random.binomial(1, worker_acc*f_diff if worker_acc*f_diff <= 1. else 1.):
                        vote = gt[item_filter_index]
                    else:
                        vote = 1 - gt[item_filter_index]
                    votes[item_filter_index][worker_id] = [vote]
    if is_gt_genereated:
        return votes, gt
    else:
        return votes
