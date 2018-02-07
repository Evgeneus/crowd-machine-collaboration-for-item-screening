import numpy as np
import pandas as pd

from generator import generate_responses_gt
from helpers.utils import run_quiz_criteria_confm
from s_run import s_run_algorithm
from machine_ensemble import machine_ensemble


if __name__ == '__main__':
    z = 0.3
    n_papers = 1000
    fr_p_part = 0.02
    baseline_items = int(fr_p_part * n_papers)
    papers_page = 10
    criteria_power = [0.14, 0.14, 0.28, 0.42]
    criteria_difficulty = [1., 1., 1.1, 0.9]
    criteria_num = len(criteria_power)
    data = []
    select_conf = 0.95
    Nt = 5
    J = 3
    # tests_num = 50
    lr = 10

    # for tests_num in [15, 20, 30, 40, 50, 100, 150, 200, 500]:
    # for lr in [1, 5, 10, 20, 50, 100]:
    for tests_num in [50]:
        # for select_conf in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]:
        # for expert_cost in [10,  20, 30, 40, 50, 70, 100]:
        for expert_cost in [20]:
            for corr in [0., 0.2, 0.3, 0.5, 0.7, 0.9]:
                print('Corr: {}, test_num: {}, baseline_items: {}, lr: {},'
                      ' select_conf: {}, expert_vote_cost: {}'.
                      format(corr, tests_num, baseline_items, lr, select_conf, expert_cost))
                loss_me_list = []
                rec_me, pre_me, f_me, f_me = [], [], [], []

                loss_smrun_list = []
                cost_smrun_list = []
                rec_sm, pre_sm, f_sm, f_sm = [], [], [], []
                loss_h_list = []
                cost_h_list = []
                rec_h, pre_h, f_h, f_h = [], [], [], []

                params = {
                    'criteria_num': criteria_num,
                    'n_papers': n_papers,
                    'papers_page': papers_page,
                    'criteria_power': criteria_power,
                    'criteria_difficulty': criteria_difficulty,
                    'fr_p_part': fr_p_part,
                    'J': J,
                    'Nt': Nt,
                    'lr': lr,
                    'corr': corr,
                    'tests_num': tests_num,
                    'select_conf': select_conf,
                    'expert_cost': expert_cost
                }

                for _ in range(50):
                    # quiz, generation responses
                    workers_accuracy = run_quiz_criteria_confm(Nt, z, [1.])
                    responses, ground_truth = generate_responses_gt(n_papers, criteria_power, papers_page,
                                                          J, workers_accuracy, criteria_difficulty)

                    params.update({
                        'ground_truth': ground_truth,
                        'workers_accuracy': workers_accuracy
                    })

                    # machine ensemble
                    loss_me, rec_me_, pre_me_, f_beta_me, prior_prob_in = machine_ensemble(params)
                    loss_me_list.append(loss_me)
                    rec_me.append(rec_me_)
                    pre_me.append(pre_me_)
                    f_me.append(f_beta_me)

                    # s-run with machine prior
                    params['prior_prob_in'] = prior_prob_in

                    loss_h, cost_h, rec_h_, pre_h_, f_beta_h = s_run_algorithm(params)
                    loss_h_list.append(loss_h)
                    cost_h_list.append(cost_h)
                    rec_h.append(rec_h_)
                    pre_h.append(pre_h_)
                    f_h.append(f_beta_h)

                    # s-run
                    loss_smrun, cost_smrun, rec_sm_, pre_sm_, f_beta_sm = s_run_algorithm(params)
                    loss_smrun_list.append(loss_smrun)
                    cost_smrun_list.append(cost_smrun)
                    rec_sm.append(rec_sm_)
                    pre_sm.append(pre_sm_)
                    f_sm.append(f_beta_sm)

                # print results
                print('ME-RUN    loss: {:1.3f}, loss_std: {:1.3f}, ' \
                      'recall: {:1.2f}, rec_std: {:1.3f}, precision: {:1.3f}, f_b: {}'. \
                      format(np.mean(loss_me_list), np.std(loss_me_list),
                             np.mean(rec_me), np.std(rec_me), np.mean(pre_me), np.mean(f_me)))

                print('SM-RUN    loss: {:1.3f}, loss_std: {:1.3f}, ' 'recall: {:1.2f}, rec_std: {:1.3f}, '
                      'price: {:1.2f}, price_std: {:1.2f}, precision: {:1.3f}, f_b: {}'
                      .format(np.mean(loss_smrun_list), np.std(loss_smrun_list), np.mean(rec_sm),
                              np.std(rec_sm), np.mean(cost_smrun_list), np.std(cost_smrun_list),
                              np.mean(pre_sm), np.mean(f_sm)))

                print('H-RUN    loss: {:1.3f}, loss_std: {:1.3f}, ' 'recall: {:1.2f}, rec_std: {:1.3f}, '
                      'price: {:1.2f}, price_std: {:1.2f}, precision: {:1.3f}, f_b: {}'
                      .format(np.mean(loss_h_list), np.std(loss_h_list), np.mean(rec_h), np.std(rec_h),
                              np.mean(cost_h_list), np.std(cost_h_list), np.mean(pre_h), np.mean(f_h)))
                print('---------------------')

                data.append([Nt, J, lr, np.mean(loss_me_list), np.std(loss_me_list), 0., 0., 'Machines-Ensemble',
                             np.mean(rec_me), np.std(rec_me), np.mean(pre_me), np.std(pre_me), np.mean(f_me),
                             np.std(f_me), tests_num, corr, select_conf, baseline_items, n_papers, expert_cost])
                data.append([Nt, J, lr, np.mean(loss_h_list), np.std(loss_h_list),
                             np.mean(cost_h_list), np.std(cost_h_list), 'Hybrid-Ensemble', np.mean(rec_h),
                             np.std(rec_h), np.mean(pre_h), np.std(pre_h), np.mean(f_h), np.std(f_h),
                             tests_num, corr, select_conf, baseline_items, n_papers, expert_cost])
                data.append([Nt, J, lr, np.mean(loss_smrun_list), np.std(loss_smrun_list),
                             np.mean(cost_smrun_list), np.std(cost_smrun_list), 'Crowd-Ensemble', np.mean(rec_sm),
                             np.std(rec_sm), np.mean(pre_sm), np.std(pre_sm), np.mean(f_sm), np.std(f_sm),
                             tests_num, corr, select_conf, baseline_items, n_papers, expert_cost])
    pd.DataFrame(data,
                 columns=['Nt', 'J', 'lr', 'loss_mean', 'loss_std', 'price_mean', 'price_std',
                          'algorithm', 'recall', 'recall_std', 'precision', 'precision_std',
                          'f_beta', 'f_beta_std', 'tests_num', 'corr', 'select_conf', 'baseline_items',
                          'total_items', 'expert_cost']
                 ).to_csv('output/data/fig0_base_settings.csv', index=False)
