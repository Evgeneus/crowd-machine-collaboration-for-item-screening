from generator import generate_responses_gt
from helpers.s_run_utils import classify_papers_baseline, generate_responses, \
    update_v_count, assign_criteria, classify_papers, update_cr_power
from helpers.utils import compute_metrics, estimate_cr_power_dif
from fusion_algorithms.algorithms_utils import input_adapter
from fusion_algorithms.em import expectation_maximization


def do_baseline_round(items_num, filters_num, papers_worker, votes_per_item, lr, ground_truth,
                   filters_select, workers_accuracy, filters_dif, values_count):
    # generate responses
    responses = generate_responses_gt(items_num, filters_select, papers_worker,
                                      votes_per_item, workers_accuracy, filters_dif, ground_truth)
    # aggregate responses
    Psi = input_adapter(responses)
    N = (items_num // papers_worker) * votes_per_item
    p = expectation_maximization(N, items_num * filters_num, Psi)[1]
    values_prob = []
    for e in p:
        e_prob = [0., 0.]
        for e_id, e_p in e.items():
            e_prob[e_id] = e_p
        values_prob.append(e_prob)

    power_cr_list, acc_cr_list = estimate_cr_power_dif(responses, filters_num, items_num, papers_worker, votes_per_item)
    classified_papers, rest_p_ids = classify_papers_baseline(range(items_num), filters_num, values_prob, lr)
    # count value counts
    for key in range(items_num*filters_num):
        cr_resp = responses[key]
        for v in cr_resp.values():
            values_count[key][v[0]] += 1
    return classified_papers, rest_p_ids, power_cr_list, acc_cr_list


def do_round(ground_truth, papers_ids, filters_num, papers_worker, workers_accuracy, filters_dif, cr_assigned):
    # generate responses
    n = len(papers_ids)
    papers_ids_rest1 = papers_ids[:n - n % papers_worker]
    papers_ids_rest2 = papers_ids[n - n % papers_worker:]
    responses_rest1 = generate_responses(ground_truth, papers_ids_rest1, filters_num,
                                         papers_worker, workers_accuracy, filters_dif,
                                         cr_assigned)
    responses_rest2 = generate_responses(ground_truth, papers_ids_rest2, filters_num,
                                         papers_worker, workers_accuracy, filters_dif,
                                         cr_assigned)
    responses = responses_rest1 + responses_rest2
    return responses


def s_run_algorithm(params):
    filters_num = params['filters_num']
    items_num = params['items_num']
    papers_worker = params['items_per_worker']
    votes_per_item = params['votes_per_item']
    lr = params['lr']
    worker_tests = params['worker_tests']
    workers_accuracy = params['workers_accuracy']
    filters_select = params['filters_select']
    filters_dif = params['filters_dif']
    ground_truth = params['ground_truth']
    fr_p_part = params['fr_p_part']
    expert_cost = params['expert_cost']
    prior_prob_in = params.get('prior_prob_in')


    # initialization
    p_thrs = 0.99
    values_count = [[0, 0] for _ in range(items_num*filters_num)]

    # Baseline round
    # in% papers
    fr_n_papers = int(items_num * fr_p_part)
    criteria_count = (worker_tests + papers_worker * filters_num) * votes_per_item * fr_n_papers // papers_worker
    first_round_res = do_baseline_round(fr_n_papers, filters_num, papers_worker, votes_per_item, lr, ground_truth,
                                     filters_select, workers_accuracy, filters_dif, values_count)
    classified_papers_fr, rest_p_ids, power_cr_list, acc_cr_list = first_round_res
    for cr_id, cr_acc in enumerate(acc_cr_list):
        if cr_acc > 0.98:
            acc_cr_list[cr_id] = 0.95
    classified_papers = dict(zip(range(items_num), [1]*items_num))
    classified_papers.update(classified_papers_fr)
    rest_p_ids = rest_p_ids + list(range(fr_n_papers, items_num))

    # compute prior power
    if prior_prob_in:
        power_cr_list = [0.] * filters_num
        for p_id in range(items_num):
            for cr in range(filters_num):
                power_cr_list[cr] += 1 - prior_prob_in[p_id * filters_num + cr]
        power_cr_list = list(map(lambda x: x / items_num, power_cr_list))

    # Do Multi rounds
    while len(rest_p_ids) != 0:

        criteria_count += len(rest_p_ids)
        cr_assigned, in_papers_ids, rest_p_ids = assign_criteria(rest_p_ids, filters_num, values_count,
                                                                 power_cr_list, acc_cr_list, prior_prob_in)

        for i in in_papers_ids:
            classified_papers[i] = 1
        responses = do_round(ground_truth, rest_p_ids, filters_num, papers_worker*filters_num,
                             workers_accuracy, filters_dif, cr_assigned)
        # update values_count
        update_v_count(values_count, filters_num, cr_assigned, responses, rest_p_ids)

        # classify papers
        classified_p_round, rest_p_ids = classify_papers(rest_p_ids, filters_num, values_count,
                                                         p_thrs, acc_cr_list, power_cr_list)

        # update criteria power
        power_cr_list = update_cr_power(items_num, filters_num, acc_cr_list, power_cr_list, values_count)

        classified_papers.update(classified_p_round)
    classified_papers = [classified_papers[p_id] for p_id in sorted(classified_papers.keys())]
    loss, recall, precision, f_beta, fp = compute_metrics(classified_papers, ground_truth, lr, filters_num)
    price_per_paper = (criteria_count + fp * expert_cost) / items_num
    return loss, price_per_paper, recall, precision, f_beta
