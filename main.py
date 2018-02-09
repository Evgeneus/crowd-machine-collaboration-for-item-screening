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
    data = []
    select_conf = 0.95
    Nt = 5
    J = 3
    tests_num = 50
    lr = 10
    expert_cost = 20
    iter_num = 50

    # criteria_num = 4
    theta = 0.3

    # for theta in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
    for criteria_num in [1, 2, 3, 4, 5, 10]:
        pow = 1 - theta**(1/criteria_num)
        criteria_power = [pow]*criteria_num
        criteria_difficulty = [1.]*criteria_num

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
            'expert_cost': expert_cost
        }

        # S-run algorithm
        loss_smrun_list = []
        cost_smrun_list = []
        rec_sm, pre_sm, f_sm, f_sm = [], [], [], []
        for _ in range(iter_num):
            # quiz, generation responses
            workers_accuracy = run_quiz_criteria_confm(Nt, z, [1.])
            responses, ground_truth = generate_responses_gt(n_papers, criteria_power, papers_page,
                                                            J, workers_accuracy, criteria_difficulty)

            params.update({
                'ground_truth': ground_truth,
                'workers_accuracy': workers_accuracy,
            })

            # s-run
            loss_smrun, cost_smrun, rec_sm_, pre_sm_, f_beta_sm = s_run_algorithm(params)
            loss_smrun_list.append(loss_smrun)
            cost_smrun_list.append(cost_smrun)
            rec_sm.append(rec_sm_)
            pre_sm.append(pre_sm_)
            f_sm.append(f_beta_sm)

        data.append([Nt, J, lr, np.mean(loss_smrun_list), np.std(loss_smrun_list),
                     np.mean(cost_smrun_list), np.std(cost_smrun_list), 'Crowd-Ensemble', np.mean(rec_sm),
                     np.std(rec_sm), np.mean(pre_sm), np.std(pre_sm), np.mean(f_sm), np.std(f_sm),
                     0., 0., select_conf, baseline_items, n_papers, expert_cost, theta, criteria_num])

        print('SM-RUN    loss: {:1.3f}, loss_std: {:1.3f}, recall: {:1.2f}, rec_std: {:1.3f}, '
              'price: {:1.2f}, price_std: {:1.2f}, precision: {:1.3f}, f_b: {}'
              .format(np.mean(loss_smrun_list), np.std(loss_smrun_list), np.mean(rec_sm),
                      np.std(rec_sm), np.mean(cost_smrun_list), np.std(cost_smrun_list),
                      np.mean(pre_sm), np.mean(f_sm)))

        # Machine and Hybrid algorithms
        for corr in [0., 0.2, 0.3, 0.5, 0.7, 0.9]:
            print('Theta: {}, filters_num: {}, Corr: {}, test_num: {}, baseline_items: {}, lr: {},'
                  ' select_conf: {}, expert_vote_cost: {}'.
                  format(theta, criteria_num, corr, tests_num, baseline_items, lr, select_conf, expert_cost))
            loss_me_list = []
            rec_me, pre_me, f_me, f_me = [], [], [], []

            loss_h_list = []
            cost_h_list = []
            rec_h, pre_h, f_h, f_h = [], [], [], []

            for _ in range(iter_num):
                # quiz, generation responses
                workers_accuracy = run_quiz_criteria_confm(Nt, z, [1.])
                responses, ground_truth = generate_responses_gt(n_papers, criteria_power, papers_page,
                                                      J, workers_accuracy, criteria_difficulty)

                params.update({
                    'corr': corr,
                    'tests_num': tests_num,
                    'select_conf': select_conf,
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

            # print results
            print('ME-RUN    loss: {:1.3f}, loss_std: {:1.3f}, recall: {:1.2f}, rec_std: {:1.3f}, '
                  'precision: {:1.3f}, f_b: {}'
                  .format(np.mean(loss_me_list), np.std(loss_me_list), np.mean(rec_me),
                          np.std(rec_me), np.mean(pre_me), np.mean(f_me)))

            print('H-RUN     loss: {:1.3f}, loss_std: {:1.3f}, ' 'recall: {:1.2f}, rec_std: {:1.3f}, '
                  'price: {:1.2f}, price_std: {:1.2f}, precision: {:1.3f}, f_b: {}'
                  .format(np.mean(loss_h_list), np.std(loss_h_list), np.mean(rec_h), np.std(rec_h),
                          np.mean(cost_h_list), np.std(cost_h_list), np.mean(pre_h), np.mean(f_h)))
            print('---------------------')

            data.append([Nt, J, lr, np.mean(loss_me_list), np.std(loss_me_list), 0.,
                         0., 'Machines-Ensemble', np.mean(rec_me), np.std(rec_me),
                         np.mean(pre_me), np.std(pre_me), np.mean(f_me), np.std(f_me), tests_num, corr,
                         select_conf, baseline_items, n_papers, expert_cost, theta, criteria_num])

            data.append([Nt, J, lr, np.mean(loss_h_list), np.std(loss_h_list),
                         np.mean(cost_h_list), np.std(cost_h_list), 'Hybrid-Ensemble', np.mean(rec_h),
                         np.std(rec_h), np.mean(pre_h), np.std(pre_h), np.mean(f_h), np.std(f_h),
                         tests_num, corr, select_conf, baseline_items, n_papers, expert_cost,
                         theta, criteria_num])

    pd.DataFrame(data,
                 columns=['Nt', 'J', 'lr', 'loss_mean', 'loss_std', 'price_mean', 'price_std',
                          'algorithm', 'recall', 'recall_std', 'precision', 'precision_std',
                          'f_beta', 'f_beta_std', 'tests_num', 'corr', 'select_conf', 'baseline_items',
                          'total_items', 'expert_cost', 'theta', 'filetrs_num']
                 ).to_csv('output/data/1/fig0_base_settings.csv', index=False)
