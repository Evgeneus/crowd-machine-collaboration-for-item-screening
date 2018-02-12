import numpy as np


def generate_gold_data(items_num, filters_select):
    gold_data = []
    for paper_id in range(items_num):
        for e_power in filters_select:
            if np.random.binomial(1, e_power):
                e_val = 1
            else:
                e_val = 0
            gold_data.append(e_val)
    return gold_data


def generate_responses_gt(items_num, filters_select, items_per_worker, worker_tests, acc, filters_dif, GT=None):
    if not GT:
        GT = generate_gold_data(items_num, filters_select)
        is_GT_genereated = True
    else:
        is_GT_genereated = False
    acc_out_list = acc[0]
    acc_in_list = acc[1]

    # generate responses
    pages_n = items_num // items_per_worker
    filters_num = len(filters_select)
    responses = {}
    for e_paper_id in range(pages_n*items_per_worker*filters_num):
        responses[e_paper_id] = {}
    for page_id in range(pages_n):
        for i in range(worker_tests):
            worker_id = page_id * worker_tests + i
            worker_acc_in = acc_in_list.pop()
            acc[1].insert(0, worker_acc_in)
            worker_acc_out = acc_out_list.pop()
            acc[0].insert(0, worker_acc_out)
            for paper_id in range(page_id * items_per_worker, page_id * items_per_worker + items_per_worker, 1):
                criteria_vals_id = range(paper_id * filters_num, paper_id * filters_num + filters_num, 1)
                isPaperIN = sum([GT[i] for i in criteria_vals_id]) == 0
                if isPaperIN:
                    worker_acc = worker_acc_in
                else:
                    worker_acc = worker_acc_out
                for e_paper_id, e_dif in zip(criteria_vals_id, filters_dif):
                    if np.random.binomial(1, worker_acc * e_dif if worker_acc * e_dif <= 1. else 1.):
                        vote = GT[e_paper_id]
                    else:
                        vote = 1 - GT[e_paper_id]
                    responses[e_paper_id][worker_id] = [vote]
    if is_GT_genereated:
        return responses, GT
    else:
        return responses
