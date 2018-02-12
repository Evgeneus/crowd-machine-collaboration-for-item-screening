import numpy as np


def generate_gold_data(items_num, criteria_power):
    gold_data = []
    for paper_id in range(items_num):
        for e_power in criteria_power:
            if np.random.binomial(1, e_power):
                e_val = 1
            else:
                e_val = 0
            gold_data.append(e_val)
    return gold_data


def generate_responses_gt(items_num, criteria_power, items_per_worker, J, acc, criteria_difficulty, GT=None):
    if not GT:
        GT = generate_gold_data(items_num, criteria_power)
        is_GT_genereated = True
    else:
        is_GT_genereated = False
    acc_out_list = acc[0]
    acc_in_list = acc[1]

    # generate responses
    pages_n = items_num // items_per_worker
    criteria_num = len(criteria_power)
    responses = {}
    for e_paper_id in range(pages_n*items_per_worker*criteria_num):
        responses[e_paper_id] = {}
    for page_id in range(pages_n):
        for i in range(J):
            worker_id = page_id * J + i
            worker_acc_in = acc_in_list.pop()
            acc[1].insert(0, worker_acc_in)
            worker_acc_out = acc_out_list.pop()
            acc[0].insert(0, worker_acc_out)
            for paper_id in range(page_id * items_per_worker, page_id * items_per_worker + items_per_worker, 1):
                criteria_vals_id = range(paper_id * criteria_num, paper_id * criteria_num + criteria_num, 1)
                isPaperIN = sum([GT[i] for i in criteria_vals_id]) == 0
                if isPaperIN:
                    worker_acc = worker_acc_in
                else:
                    worker_acc = worker_acc_out
                for e_paper_id, e_dif in zip(criteria_vals_id, criteria_difficulty):
                    if np.random.binomial(1, worker_acc * e_dif if worker_acc * e_dif <= 1. else 1.):
                        vote = GT[e_paper_id]
                    else:
                        vote = 1 - GT[e_paper_id]
                    responses[e_paper_id][worker_id] = [vote]
    if is_GT_genereated:
        return responses, GT
    else:
        return responses
