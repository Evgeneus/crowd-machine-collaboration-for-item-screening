from generator import generate_votes_gt
from helpers.s_run_utils import classify_items_baseround, generate_votes, \
    update_v_count, assign_criteria, classify_papers, update_cr_power
from helpers.utils import compute_metrics, estimate_filters_property
from fusion_algorithms.algorithms_utils import input_adapter
from fusion_algorithms.em import expectation_maximization


def do_baseround(items_num, filters_num, items_per_worker, votes_per_item, ground_truth,
                   filters_select, workers_accuracy, filters_dif, values_count):
    # generate votes
    votes = generate_votes_gt(items_num, filters_select, items_per_worker,
                                      votes_per_item, workers_accuracy, filters_dif, ground_truth)
    # aggregate votes via truth finder
    Psi = input_adapter(votes)
    N = (items_num // items_per_worker) * votes_per_item
    _, p = expectation_maximization(N, items_num * filters_num, Psi)
    values_prob = []
    for e in p:
        e_prob = [0., 0.]
        for e_id, e_p in e.items():
            e_prob[e_id] = e_p
        values_prob.append(e_prob)

    filters_select_est, filters_dif_est = estimate_filters_property(votes, filters_num, items_num, items_per_worker, votes_per_item)
    items_classified, items_to_classify = classify_items_baseround(range(items_num), filters_num, values_prob)
    # count value counts
    for key in range(items_num*filters_num):
        filter_item_votes = votes[key]
        for v in filter_item_votes.values():
            values_count[key][v[0]] += 1
    return items_classified, items_to_classify, filters_select_est, filters_dif_est


def do_round(ground_truth, papers_ids, filters_num, items_per_worker, workers_accuracy, filters_dif, cr_assigned):
    # generate votes
    n = len(papers_ids)
    papers_ids_rest1 = papers_ids[:n - n % items_per_worker]
    papers_ids_rest2 = papers_ids[n - n % items_per_worker:]
    votes_rest1 = generate_votes(ground_truth, papers_ids_rest1, filters_num,
                                         items_per_worker, workers_accuracy, filters_dif,
                                         cr_assigned)
    votes_rest2 = generate_votes(ground_truth, papers_ids_rest2, filters_num,
                                         items_per_worker, workers_accuracy, filters_dif,
                                         cr_assigned)
    votes = votes_rest1 + votes_rest2
    return votes


def s_run_algorithm(params):
    filters_num = params['filters_num']
    items_num = params['items_num']
    items_per_worker = params['items_per_worker']
    votes_per_item = params['votes_per_item']
    lr = params['lr']
    worker_tests = params['worker_tests']
    workers_accuracy = params['workers_accuracy']
    filters_select = params['filters_select']
    filters_dif = params['filters_dif']
    ground_truth = params['ground_truth']
    baseround_items = params['baseround_items']
    expert_cost = params['expert_cost']
    prior_prob_in = params.get('prior_prob_in')


    # initialization
    p_thrs = 0.99
    values_count = [[0, 0] for _ in range(items_num*filters_num)]

    # base round
    # in% papers
    criteria_count = (worker_tests + items_per_worker*filters_num) * votes_per_item * baseround_items // items_per_worker
    baseround_res = do_baseround(baseround_items, filters_num, items_per_worker, votes_per_item, ground_truth,
                                        filters_select, workers_accuracy, filters_dif, values_count)
    items_classified_fr, items_to_classify, power_cr_list, acc_cr_list = baseround_res
    for cr_id, cr_acc in enumerate(acc_cr_list):
        if cr_acc > 0.98:
            acc_cr_list[cr_id] = 0.95
    items_classified = dict(zip(range(items_num), [1]*items_num))
    items_classified.update(items_classified_fr)
    items_to_classify = items_to_classify + list(range(baseround_items, items_num))

    # compute prior power
    if prior_prob_in:
        power_cr_list = [0.] * filters_num
        for p_id in range(items_num):
            for cr in range(filters_num):
                power_cr_list[cr] += 1 - prior_prob_in[p_id * filters_num + cr]
        power_cr_list = list(map(lambda x: x / items_num, power_cr_list))

    # Do Multi rounds
    while len(items_to_classify) != 0:

        criteria_count += len(items_to_classify)
        cr_assigned, in_papers_ids, items_to_classify = assign_criteria(items_to_classify, filters_num, values_count,
                                                                 power_cr_list, acc_cr_list, prior_prob_in)

        for i in in_papers_ids:
            items_classified[i] = 1
        votes = do_round(ground_truth, items_to_classify, filters_num, items_per_worker*filters_num,
                             workers_accuracy, filters_dif, cr_assigned)
        # update values_count
        update_v_count(values_count, filters_num, cr_assigned, votes, items_to_classify)

        # classify papers
        classified_p_round, items_to_classify = classify_papers(items_to_classify, filters_num, values_count,
                                                         p_thrs, acc_cr_list, power_cr_list)

        # update criteria power
        power_cr_list = update_cr_power(items_num, filters_num, acc_cr_list, power_cr_list, values_count)

        items_classified.update(classified_p_round)
    items_classified = [items_classified[p_id] for p_id in sorted(items_classified.keys())]
    loss, recall, precision, f_beta, fp = compute_metrics(items_classified, ground_truth, lr, filters_num)
    price_per_paper = (criteria_count + fp * expert_cost) / items_num
    return loss, price_per_paper, recall, precision, f_beta
