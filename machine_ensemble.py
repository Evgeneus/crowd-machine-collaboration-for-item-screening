import numpy as np
from helpers.utils import compute_metrics
from scipy.stats import beta


def generate_vote(gt, acc, corr, vote_prev):
    if np.random.binomial(1, corr, 1)[0]:
        vote = vote_prev
    else:
        if np.random.binomial(1, acc):
            vote = gt
        else:
            vote = 1 - gt
    return vote


def test_machines_accuracy(filters_num, machines_params, tests_num):
    # test machines
    machines_accuracy = [[] for _ in range(filters_num)]
    for filter_index in range(filters_num):
        for machine in machines_params[filter_index]:
            # Indeed,  we need to reject the machine if acc < 0.5
            while True:
                correct_votes_num = sum(np.random.binomial(1, machine[0], tests_num // 2)) + \
                                    sum(np.random.binomial(1, machine[1], tests_num // 2))
                acc = correct_votes_num / tests_num
                if acc > 0.5 and acc != 1.:
                    break
            machines_accuracy[filter_index].append(acc)
    return machines_accuracy


# fuse votes via weighted majority voting
# output: probabilities to be negatives for each filter and item
def weighted_mv(votes_list, filters_num, items_num, machines_accuracy):
    probs_list = [None]*filters_num*items_num
    for filter_index in range(filters_num):
        filter_machines_acc = machines_accuracy
        for item_index in range(items_num):
            like_true_val = 1  # assume true value is positive
            a, b = 1., 1.  # constituents of baysian formula, prior is uniform dist.
            # a responds for positives, b - for negatives
            for vote, acc in zip(votes_list[item_index*filters_num + filter_index], filter_machines_acc):
                if vote == like_true_val:
                    a *= acc
                    b *= 1 - acc
                else:
                    a *= 1 - acc
                    b *= acc
            probs_list[item_index*filters_num + filter_index] = b / (a + b)
    return probs_list


def classify_items(ensembled_votes, lr, filters_num, items_num):
    prob_in_list = []
    items_labels = []
    # pos_thr = lr / (1. + lr)  # threshold to classify as a positive
    pos_thr = 0.99  # threshold to classify as a positive
    for item_index in range(items_num):
        prob_all_not_apply = 1.
        for filter_index in range(filters_num):
            prob_all_not_apply *= ensembled_votes[item_index*filters_num + filter_index]
        prob_item_out = 1. - prob_all_not_apply
        prob_in_list.append(prob_all_not_apply)

        # classify item
        if prob_item_out > pos_thr:
            items_labels.append(0)
        else:
            items_labels.append(1)
    return items_labels, prob_in_list


def get_machines(corr, test_num, machine_selec_conf):
    machines_num = 10
    first_machine_acc = np.random.uniform(0.55, 0.9)
    # print("first_machine_acc: {}".format(first_machine_acc))
    test_votes = [[] for _ in range(machines_num)]
    test_votes[0] = list(np.random.binomial(1, first_machine_acc, test_num))

    machines_acc = [first_machine_acc] + list(np.random.uniform(0.5, 0.95, machines_num - 1))
    for m_id, acc in enumerate(machines_acc[1:]):
        for i in range(test_num):
            if np.random.binomial(1, corr, 1)[0]:
                vote = test_votes[0][i]
            else:
                vote = np.random.binomial(1, acc, 1)[0]
            test_votes[m_id+1].append(vote)

    selected_machines_acc = []
    for machine_votes, acc in zip(test_votes, machines_acc):
        correct_votes_num = sum(machine_votes)
        conf = beta.sf(0.5, correct_votes_num+1, test_num-correct_votes_num+1)
        if conf > machine_selec_conf:
            selected_machines_acc.append(acc)

    # check number of machines passed the tests
    # add at least one machine with accuracy in [0.55, 0.9]
    if len(selected_machines_acc) < 1:
        selected_machines_acc.append(np.random.uniform(0.55, 0.9))

    return selected_machines_acc


def machine_ensemble(filters_num, items_num, gt_values, lr, corr, test_num, machine_selec_conf):
    # parameters for machine-based classifiers (accuracy for positives, accuracy for negatives)
    # positive vote - out of scope
    # negative vote - in scope
    # machines_params = [[(0.9, 0.8), (0.9, 0.8), (0.9, 0.8), (0.9, 0.8), (0.9, 0.8)],  # machines for criteria 0
    #                    [(0.9, 0.8), (0.9, 0.8), (0.9, 0.8), (0.9, 0.8), (0.9, 0.8)],  # machines for criteria 1
    #                    [(0.9, 0.8), (0.9, 0.8), (0.9, 0.8), (0.9, 0.8), (0.9, 0.8)],  # machines for criteria 2
    #                    [(0.9, 0.8), (0.9, 0.8), (0.9, 0.8), (0.9, 0.8), (0.9, 0.8)]]  # machines for criteria 3
    machines_accuracy = get_machines(corr, test_num, machine_selec_conf)

    votes_list = [[] for _ in range(items_num*filters_num)]

    for item_index in range(items_num):
        for filter_index in range(filters_num):
            gt = gt_values[item_index*filters_num + filter_index]  # can be either 0 or 1
            if np.random.binomial(1, machines_accuracy[0]):
                vote = gt
            else:
                vote = 1 - gt
            votes_list[item_index * filters_num + filter_index].append(vote)

    for item_index in range(items_num):
        for filter_index in range(filters_num):
            gt = gt_values[item_index*filters_num + filter_index]  # can be either 0 or 1
            vote_prev = votes_list[item_index*filters_num + filter_index][0]
            for machine in machines_accuracy[1:]:
                vote = generate_vote(gt, machine, corr, vote_prev)
                votes_list[item_index*filters_num + filter_index].append(vote)

    # # estimate machines accuracies
    # machines_accuracy = test_machines_accuracy(filters_num, machines_params)
    # ensemble votes for each filter and item
    ensembled_votes_in = weighted_mv(votes_list, filters_num, items_num, machines_accuracy)
    items_labels, prob_in_list = classify_items(ensembled_votes_in, lr, filters_num, items_num)
    loss, fp_rate, fn_rate, recall, precision, f_beta = compute_metrics(items_labels, gt_values, lr, filters_num)
    return loss, fp_rate, fn_rate, recall, precision, f_beta, ensembled_votes_in

