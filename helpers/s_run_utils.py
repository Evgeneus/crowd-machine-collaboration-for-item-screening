import numpy as np
from scipy.special import binom


def assign_criteria(papers_ids, filters_num, values_count, power_cr_list, filters_acc_est, prior_prob_in=None):
    cr_assigned = []
    papers_ids_new = []
    cr_list = range(filters_num)
    # in_papers_ids = []
    for item_index in papers_ids:
        p_classify = []
        n_min_list = []
        joint_prob_votes_out = [1.]*filters_num
        for cr in cr_list:
            acc_cr = filters_acc_est[cr]
            if prior_prob_in != None:
                p_paper_out = 1 - prior_prob_in[item_index * filters_num + cr]
            else:
                p_paper_out = power_cr_list[cr]
            cr_count = values_count[item_index * filters_num + cr]
            in_c = cr_count[0]
            out_c = cr_count[1]
            for n in range(1, 11):
                # new value is out
                p_vote_out = acc_cr * p_paper_out + (1 - acc_cr) * (1 - p_paper_out)
                joint_prob_votes_out[cr] *= p_vote_out
                term1_p_out = binom(in_c+out_c+n, out_c+n)*acc_cr**(out_c+n)*(1-acc_cr)**in_c*power_cr_list[cr]
                term1_p_in = binom(in_c+out_c+n, in_c)*acc_cr**in_c*(1-acc_cr)**(out_c+n)*(1-power_cr_list[cr])
                p_paper_in_vote_out = term1_p_in * p_vote_out / (term1_p_out + term1_p_in)
                p_paper_out = 1 - p_paper_in_vote_out
                if p_paper_out >= 0.99:
                    p_classify.append(joint_prob_votes_out[cr]/n)
                    n_min_list.append(n)
                    break
                elif n == 10:
                    p_classify.append(joint_prob_votes_out[cr]/n)
                    n_min_list.append(n)
        cr_assign = p_classify.index(max(p_classify))
        n_min = n_min_list[cr_assign]
        joint_prob = joint_prob_votes_out[cr_assign]

        # # check stopping condition
        # if n_min / joint_prob >= 15:
        #     in_papers_ids.append(item_index)
        # else:
        #     cr_assigned.append(cr_assign)
        #     papers_ids_new.append(item_index)
        # check stopping condition
        if n_min / joint_prob < 15:
            cr_assigned.append(cr_assign)
            papers_ids_new.append(item_index)
    return cr_assigned, papers_ids_new


def classify_items_baseround(papers_ids, filters_num, values_prob):
    classified_papers = []
    classified_papers_ids = []
    rest_papers_ids = []
    trsh = 0.99
    for paper_id in papers_ids:
        p_inclusion = 1.
        for e_paper_id in range(filters_num):
            p_inclusion *= values_prob[paper_id * filters_num + e_paper_id][0]
        p_exclusion = 1 - p_inclusion

        if p_exclusion > trsh:
            classified_papers.append(0)
            classified_papers_ids.append(paper_id)
        elif p_inclusion > trsh:
            classified_papers.append(1)
            classified_papers_ids.append(paper_id)
        else:
            rest_papers_ids.append(paper_id)
    return dict(zip(classified_papers_ids, classified_papers)), rest_papers_ids


def classify_items(papers_ids, filters_num, values_count, p_thrs, filters_acc_est, power_cr_list, prior_prob_in=None):
    classified_papers = []
    classified_papers_ids = []
    rest_papers_ids = []

    for item_index in papers_ids:
        p_inclusion = 1.
        for cr in range(filters_num):
            acc_cr = filters_acc_est[cr]
            # power_cr = power_cr_list[cr]
            cr_count = values_count[item_index * filters_num + cr]
            in_c = cr_count[0]
            out_c = cr_count[1]
            if prior_prob_in != None:
                p_paper_out = 1 - prior_prob_in[item_index * filters_num + cr]
            else:
                p_paper_out = power_cr_list[cr]

            if in_c == 0 and out_c == 0:
                prob_cr_in = 1 - p_paper_out
            else:
                prop_p_in = binom(in_c+out_c, in_c)*acc_cr**in_c*(1-acc_cr)**out_c*(1-p_paper_out)
                prop_p_out = binom(in_c+out_c, out_c)*acc_cr**out_c*(1-acc_cr)**in_c*p_paper_out
                prob_cr_in = prop_p_in / (prop_p_in + prop_p_out)
            p_inclusion *= prob_cr_in
        p_exclusion = 1 - p_inclusion

        if p_exclusion > p_thrs:
            classified_papers.append(0)
            classified_papers_ids.append(item_index)
        elif p_inclusion > p_thrs:
            classified_papers.append(1)
            classified_papers_ids.append(item_index)
        else:
            rest_papers_ids.append(item_index)
    return dict(zip(classified_papers_ids, classified_papers)), rest_papers_ids


def generate_votes(GT, papers_ids, filters_num, items_per_worker, acc, filters_dif, cr_assigned):
    votes = []
    n = len(papers_ids)
    workers_n = 1 if n < items_per_worker else n // items_per_worker
    for w_ind in range(workers_n):
        worker_acc_in = acc[1].pop()
        acc[1].insert(0, worker_acc_in)
        worker_acc_out = acc[0].pop()
        acc[0].insert(0, worker_acc_out)
        for cr, item_index in zip(cr_assigned[w_ind*items_per_worker: w_ind*items_per_worker+items_per_worker],
                            papers_ids[w_ind*items_per_worker: w_ind*items_per_worker+items_per_worker]):
            cr_vals_id = range(item_index * filters_num, item_index * filters_num + filters_num, 1)
            isPaperIN = sum([GT[i] for i in cr_vals_id]) == 0
            if isPaperIN:
                worker_acc = worker_acc_in
            else:
                worker_acc = worker_acc_out

            GT_cr = GT[item_index * filters_num + cr]
            cr_dif = filters_dif[cr]
            if np.random.binomial(1, worker_acc * cr_dif if worker_acc * cr_dif <= 1. else 1.):
                vote = GT_cr
            else:
                vote = 1 - GT_cr
            votes.append(vote)
    return votes


def update_v_count(values_count, filters_num, cr_assigned, votes, p_ids):
    for cr, vote, item_index in zip(cr_assigned, votes, p_ids):
        if vote:
            values_count[item_index * filters_num + cr][1] += 1
        else:
            values_count[item_index * filters_num + cr][0] += 1


def update_filters_select(items_num, filters_num, filters_acc_est, power_cr_list, values_count):
    power_cr_new = []
    apply_criteria_list = [[] for _ in range(filters_num)]
    for item_index in range(items_num):
        for cr in range(filters_num):
            acc_cr = filters_acc_est[cr]
            power_cr = power_cr_list[cr]
            cr_count = values_count[item_index * filters_num + cr]
            in_c = cr_count[0]
            out_c = cr_count[1]
            prop_p_in = binom(in_c+out_c, in_c)*acc_cr**in_c*(1-acc_cr)**out_c*(1-power_cr)
            prop_p_out = binom(in_c+out_c, out_c)*acc_cr**out_c*(1-acc_cr)**in_c*power_cr
            prob_cr_out = prop_p_out / (prop_p_in + prop_p_out)
            apply_criteria_list[cr].append(prob_cr_out)
    for probs in apply_criteria_list:
        power_cr_new.append(np.mean(probs))
    return power_cr_new
