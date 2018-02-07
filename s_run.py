from generator import generate_responses_gt
from helpers.s_run_utils import classify_papers_baseline, generate_responses, \
    update_v_count, assign_criteria, classify_papers, update_cr_power
from helpers.utils import compute_metrics, estimate_cr_power_dif
from fusion_algorithms.algorithms_utils import input_adapter
from fusion_algorithms.em import expectation_maximization


def do_first_round(n_papers, criteria_num, papers_worker, J, lr, GT,
                   criteria_power, acc, criteria_difficulty, values_count):
    # generate responses
    responses = generate_responses_gt(n_papers, criteria_power, papers_worker,
                                      J, acc, criteria_difficulty, GT)
    # aggregate responses
    Psi = input_adapter(responses)
    N = (n_papers // papers_worker) * J
    p = expectation_maximization(N, n_papers * criteria_num, Psi)[1]
    values_prob = []
    for e in p:
        e_prob = [0., 0.]
        for e_id, e_p in e.items():
            e_prob[e_id] = e_p
        values_prob.append(e_prob)

    power_cr_list, acc_cr_list = estimate_cr_power_dif(responses, criteria_num, n_papers, papers_worker, J)
    classified_papers, rest_p_ids = classify_papers_baseline(range(n_papers), criteria_num, values_prob, lr)
    # count value counts
    for key in range(n_papers*criteria_num):
        cr_resp = responses[key]
        for v in cr_resp.values():
            values_count[key][v[0]] += 1
    return classified_papers, rest_p_ids, power_cr_list, acc_cr_list


def do_round(GT, papers_ids, criteria_num, papers_worker, acc, criteria_difficulty, cr_assigned):
    # generate responses
    n = len(papers_ids)
    papers_ids_rest1 = papers_ids[:n - n % papers_worker]
    papers_ids_rest2 = papers_ids[n - n % papers_worker:]
    responses_rest1 = generate_responses(GT, papers_ids_rest1, criteria_num,
                                         papers_worker, acc, criteria_difficulty,
                                         cr_assigned)
    responses_rest2 = generate_responses(GT, papers_ids_rest2, criteria_num,
                                         papers_worker, acc, criteria_difficulty,
                                         cr_assigned)
    responses = responses_rest1 + responses_rest2
    return responses


def s_run(criteria_num, n_papers, papers_worker, J, lr, Nt, acc,
           criteria_power, criteria_difficulty, GT, fr_p_part, expert_vote_cost):
    # initialization
    p_thrs = 0.99
    values_count = [[0, 0] for _ in range(n_papers*criteria_num)]

    # Baseline round
    # in% papers
    fr_n_papers = int(n_papers * fr_p_part)
    criteria_count = (Nt + papers_worker * criteria_num) * J * fr_n_papers // papers_worker
    first_round_res = do_first_round(fr_n_papers, criteria_num, papers_worker, J, lr, GT,
                                     criteria_power, acc, criteria_difficulty, values_count)
    classified_papers_fr, rest_p_ids, power_cr_list, acc_cr_list = first_round_res
    for cr_id, cr_acc in enumerate(acc_cr_list):
        if cr_acc > 0.98:
            acc_cr_list[cr_id] = 0.95
    classified_papers = dict(zip(range(n_papers), [1]*n_papers))
    classified_papers.update(classified_papers_fr)
    rest_p_ids = rest_p_ids + list(range(fr_n_papers, n_papers))

    # Do Multi rounds
    while len(rest_p_ids) != 0:

        criteria_count += len(rest_p_ids)
        cr_assigned, in_papers_ids, rest_p_ids = assign_criteria(rest_p_ids, criteria_num, values_count, power_cr_list, acc_cr_list)

        for i in in_papers_ids:
            classified_papers[i] = 1
        responses = do_round(GT, rest_p_ids, criteria_num, papers_worker*criteria_num,
                             acc, criteria_difficulty, cr_assigned)
        # update values_count
        update_v_count(values_count, criteria_num, cr_assigned, responses, rest_p_ids)

        # classify papers
        classified_p_round, rest_p_ids = classify_papers(rest_p_ids, criteria_num, values_count,
                                                         p_thrs, acc_cr_list, power_cr_list)

        # update criteria power
        power_cr_list = update_cr_power(n_papers, criteria_num, acc_cr_list, power_cr_list, values_count)

        classified_papers.update(classified_p_round)
    classified_papers = [classified_papers[p_id] for p_id in sorted(classified_papers.keys())]
    loss, fp_rate, fn_rate, recall, precision, f_beta, fp = compute_metrics(classified_papers, GT, lr, criteria_num)
    # price_per_paper = criteria_count / n_papers
    price_per_paper = (criteria_count + fp * expert_vote_cost) / n_papers
    return loss, price_per_paper, fp_rate, fn_rate, recall, precision, f_beta
