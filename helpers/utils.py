from quiz_simulation import do_quiz_criteria_confm
from fusion_algorithms.algorithms_utils import input_adapter
from fusion_algorithms.em import expectation_maximization

import numpy as np


def run_quiz_criteria_confm(quiz_papers_n, cheaters_prop, criteria_difficulty):
    acc_passed_distr = [[], []]
    for _ in range(100000):
        result = do_quiz_criteria_confm(quiz_papers_n, cheaters_prop, criteria_difficulty)
        if len(result) > 1:
            acc_passed_distr[0].append(result[0])
            acc_passed_distr[1].append(result[1])
    return acc_passed_distr


def compute_metrics(classified_papers, GT, lr, criteria_num):
    # obtain GT scope values for papers
    GT_scope = []
    for paper_id in range(len(classified_papers)):
        if sum([GT[paper_id * criteria_num + e_paper_id] for e_paper_id in range(criteria_num)]):
            GT_scope.append(0)
        else:
            GT_scope.append(1)
    # Positive == Inclusion (Relevant)
    # N == Exclusion (Not relevant)
    fp = 0.
    fn = 0.
    tp = 0.
    tn = 0.
    for cl_val, gt_val in zip(classified_papers, GT_scope):
        if gt_val and not cl_val:
            fn += 1
        if not gt_val and cl_val:
            fp += 1
        if gt_val and cl_val:
            tp += 1
        if not gt_val and not cl_val:
            tn += 1
    tp_rate = tp / (fn + tp)
    fp_rate = fp / (fp + tn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    loss = (fn * lr + fp) / len(classified_papers)
    beta = 1. / lr
    f_beta = (beta + 1) * precision * recall / (beta * recall + precision)
    return loss, fp_rate, tp_rate, recall, precision, f_beta, fp


def classify_papers(n_papers, criteria_num, responses, papers_page, J, lr):
    Psi = input_adapter(responses)
    N = n_papers // papers_page * J
    p = expectation_maximization(N, n_papers * criteria_num, Psi)[1]
    values_prob = []
    for e in p:
        e_prob = [0., 0.]
        for e_id, e_p in e.items():
            e_prob[e_id] = e_p
        values_prob.append(e_prob)

    classified_papers = []
    exclusion_trsh = lr / (lr + 1.)
    for paper_id in range(n_papers):
        p_inclusion = 1.
        for e_paper_id in range(criteria_num):
            p_inclusion *= values_prob[paper_id*criteria_num+e_paper_id][0]
        p_exclusion = 1 - p_inclusion
        classified_papers.append(0) if p_exclusion > exclusion_trsh else classified_papers.append(1)
    return classified_papers


def estimate_cr_power_dif(responses, criteria_num, n_papers, papers_page, J):
    Psi = input_adapter(responses)
    N = (n_papers // papers_page) * J
    cr_power = []
    cr_accuracy = []
    for cr in range(criteria_num):
        cr_responses = Psi[cr::criteria_num]
        acc_list, p_cr = expectation_maximization(N, n_papers, cr_responses)
        acc_cr = np.mean(acc_list)
        pow_cr = 0.
        for e in p_cr:
            e_prob = [0., 0.]
            for e_id, e_p in e.items():
                e_prob[e_id] = e_p
            pow_cr += e_prob[1]
        pow_cr /= n_papers
        cr_power.append(pow_cr)
        cr_accuracy.append(acc_cr)

    return cr_power, cr_accuracy

