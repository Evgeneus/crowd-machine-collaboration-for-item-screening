from helpers.s_run_utils import classify_items_baseround, generate_votes, \
     update_votes_count, assign_filters, classify_items, update_filters_select, \
     estimate_filters_property
from helpers.utils import compute_metrics, generate_votes_gt
from fusion_algorithms.algorithms_utils import input_adapter
from fusion_algorithms.em import expectation_maximization


def do_baseround(items_num, filters_num, items_per_worker, votes_per_item, ground_truth,
                 filters_select, workers_accuracy, filters_dif, values_count):
    # generate votes
    votes = generate_votes_gt(items_num, filters_select, items_per_worker,
                              votes_per_item, workers_accuracy, filters_dif, ground_truth)
    # aggregate votes via truth finder
    psi = input_adapter(votes)
    n = (items_num // items_per_worker) * votes_per_item
    _, p = expectation_maximization(n, items_num * filters_num, psi)
    values_prob = []
    for e in p:
        e_prob = [0., 0.]
        for e_id, e_p in e.items():
            e_prob[e_id] = e_p
        values_prob.append(e_prob)

    filters_select_est, filters_acc_est = estimate_filters_property(votes, filters_num, items_num,
                                                                    items_per_worker, votes_per_item)
    items_classified, items_to_classify = classify_items_baseround(range(items_num), filters_num, values_prob)
    # count value counts
    for key in range(items_num*filters_num):
        filter_item_votes = votes[key]
        for v in filter_item_votes.values():
            values_count[key][v[0]] += 1

    return items_classified, items_to_classify, filters_select_est, filters_acc_est


def do_round(ground_truth, items, filters_num, items_per_worker, workers_accuracy, filters_dif, filters_assigned):
    n = len(items)
    items_batch1 = items[:n - n % items_per_worker]
    items_rest2 = items[n - n % items_per_worker:]

    votes_batch1 = generate_votes(ground_truth, items_batch1, filters_num, items_per_worker,
                                  workers_accuracy, filters_dif, filters_assigned)
    votes_batch2 = generate_votes(ground_truth, items_rest2, filters_num, items_per_worker,
                                  workers_accuracy, filters_dif, filters_assigned)

    votes = votes_batch1 + votes_batch2
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
    prior_prob_pos = params.get('prior_prob_pos')

    # initialization
    p_thrs = 0.99
    values_count = [[0, 0] for _ in range(items_num*filters_num)]

    # base round
    # in% papers
    votes_count = (worker_tests + items_per_worker*filters_num) * votes_per_item * baseround_items // items_per_worker
    baseround_res = do_baseround(baseround_items, filters_num, items_per_worker, votes_per_item, ground_truth,
                                 filters_select, workers_accuracy, filters_dif, values_count)
    items_classified_baseround, items_to_classify, filters_select_est, filters_acc_est = baseround_res

    # check for bordercases
    for filter_index, filter_acc in enumerate(filters_acc_est):
        if filter_acc > 0.98:
            filters_acc_est[filter_index] = 0.95

    items_classified = dict(zip(range(items_num), [1]*items_num))
    items_classified.update(items_classified_baseround)
    items_to_classify = items_to_classify + list(range(baseround_items, items_num))

    # compute prior power
    if prior_prob_pos:
        filters_select_est = [0.] * filters_num
        for item_index in range(items_num):
            for filter_index in range(filters_num):
                filters_select_est[filter_index] += 1 - prior_prob_pos[item_index*filters_num + filter_index]
        filters_select_est = list(map(lambda x: x / items_num, filters_select_est))

    # Do Multi rounds
    while len(items_to_classify) != 0:

        votes_count += len(items_to_classify)
        filters_assigned, items_to_classify = assign_filters(items_to_classify, filters_num, values_count,
                                                             filters_select_est, filters_acc_est, prior_prob_pos)

        votes = do_round(ground_truth, items_to_classify, filters_num, items_per_worker*filters_num,
                         workers_accuracy, filters_dif, filters_assigned)
        # update values_count
        update_votes_count(values_count, filters_num, filters_assigned, votes, items_to_classify)

        # update filters selectivity
        filters_select_est = update_filters_select(items_num, filters_num, filters_acc_est,
                                                   filters_select_est, values_count)

        # classify items
        items_classified_round, items_to_classify = classify_items(items_to_classify, filters_num, values_count,
                                                                   p_thrs, filters_acc_est, filters_select_est)
        items_classified.update(items_classified_round)

    items_classified = [items_classified[item_index] for item_index in sorted(items_classified.keys())]
    loss, recall, precision, f_beta, fp = compute_metrics(items_classified, ground_truth, lr, filters_num)
    price_per_paper = (votes_count + fp*expert_cost) / items_num

    return loss, price_per_paper, recall, precision, f_beta
