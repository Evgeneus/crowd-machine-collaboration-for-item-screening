import numpy as np
import pandas as pd

from generator import generate_responses_gt
from helpers.utils import run_quiz_criteria_confm
from s_run import s_run
from machine_ensemble import machine_ensemble
from hybrid_classifier import hybrid_classifier


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
    machine_selec_conf = 0.95
    Nt = 5
    J = 3
    tests_num = 50
    lr = 5

    # for tests_num in [15, 20, 30, 40, 50, 100, 150, 200, 500]:
    # for lr in [1, 5, 10, 20, 50, 100]:
    # for tests_num in [50]:
    for machine_selec_conf in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]:
        for corr in [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            print('Corr: {}, test_num: {}, baseline_items: {}, lr: {}, machine_selec_conf: {}'.
                  format(corr, tests_num, baseline_items, lr, machine_selec_conf))
            loss_me_list = []
            fp_me, tp_me, rec_me, pre_me, f_me, f_me = [], [], [], [], [], []

            loss_smrun_list = []
            cost_smrun_list = []
            fp_sm, tp_sm, rec_sm, pre_sm, f_sm, f_sm = [], [], [], [], [], []

            loss_h_list = []
            cost_h_list = []
            fp_h, tp_h, rec_h, pre_h, f_h, f_h = [], [], [], [], [], []
            for _ in range(10):
                # quiz, generation responses
                acc = run_quiz_criteria_confm(Nt, z, [1.])
                responses, GT = generate_responses_gt(n_papers, criteria_power, papers_page,
                                                      J, acc, criteria_difficulty)

                # machine ensemble
                loss_me, fp_rate_me, tp_rate_me, \
                rec_me_, pre_me_, f_beta_me, prior_prob_in = machine_ensemble(criteria_num, n_papers, GT,
                                                                              lr, corr, tests_num, machine_selec_conf)
                loss_me_list.append(loss_me)
                fp_me.append(fp_rate_me)
                tp_me.append(tp_rate_me)
                rec_me.append(rec_me_)
                pre_me.append(pre_me_)
                f_me.append(f_beta_me)

                # hybrid classifier
                loss_h, cost_h, fp_rate_h, tp_rate_h, \
                rec_h_, pre_h_, f_beta_h = hybrid_classifier(criteria_num, n_papers, papers_page, J, lr, Nt, acc,
                                                             criteria_power, criteria_difficulty, GT, fr_p_part,
                                                             prior_prob_in)
                loss_h_list.append(loss_h)
                cost_h_list.append(cost_h)
                fp_h.append(fp_rate_h)
                tp_h.append(tp_rate_h)
                rec_h.append(rec_h_)
                pre_h.append(pre_h_)
                f_h.append(f_beta_h)

                # sm-run
                loss_smrun, cost_smrun, fp_rate_sm, tp_rate_sm, \
                rec_sm_, pre_sm_, f_beta_sm = s_run(criteria_num, n_papers, papers_page, J, lr, Nt, acc,
                                                     criteria_power, criteria_difficulty, GT, fr_p_part)
                loss_smrun_list.append(loss_smrun)
                cost_smrun_list.append(cost_smrun)
                fp_sm.append(fp_rate_sm)
                tp_sm.append(tp_rate_sm)
                rec_sm.append(rec_sm_)
                pre_sm.append(pre_sm_)
                f_sm.append(f_beta_sm)

            # print results
            print('ME-RUN    loss: {:1.3f}, loss_std: {:1.3f}, ' \
                  'recall: {:1.2f}, precision: {:1.2f}, f_b: {}'. \
                  format(np.mean(loss_me_list), np.std(loss_me_list),
                         np.mean(rec_me), np.mean(pre_me), np.mean(f_me)))

            print('SM-RUN    loss: {:1.3f}, loss_std: {:1.3f}, price: {:1.2f}, ' \
                  'recall: {:1.2f}, precision: {:1.2f}, f_b: {}'. \
                  format(np.mean(loss_smrun_list), np.std(loss_smrun_list), np.mean(cost_smrun_list),
                         np.mean(rec_sm), np.mean(pre_sm), np.mean(f_sm)))

            print('H-RUN    loss: {:1.3f}, loss_std: {:1.3f},  price: {:1.2f}, ' \
                  'recall: {:1.2f}, precision: {:1.2f}, f_b: {}'. \
                  format(np.mean(loss_h_list), np.std(loss_h_list),  np.mean(cost_h_list),
                         np.mean(rec_h), np.mean(pre_h), np.mean(f_h)))
            print('---------------------')

            data.append([Nt, J, lr, np.mean(loss_me_list), np.std(loss_me_list),
                         0., 0., 'Machines-Ensemble', np.mean(rec_me), np.mean(pre_me),
                         np.mean(f_me), tests_num, corr, machine_selec_conf, baseline_items, n_papers])
            data.append([Nt, J, lr, np.mean(loss_h_list), np.std(loss_h_list),
                         np.mean(cost_h_list), np.std(cost_h_list), 'Hybrid-Ensemble',
                         np.mean(rec_h), np.mean(pre_h), np.mean(f_h), tests_num, corr,
                         machine_selec_conf, baseline_items, n_papers])
            data.append([Nt, J, lr, np.mean(loss_smrun_list), np.std(loss_smrun_list),
                         np.mean(cost_smrun_list), np.std(cost_smrun_list), 'Crowd-Ensemble',
                         np.mean(rec_sm), np.mean(pre_sm), np.mean(f_sm), tests_num, corr,
                         machine_selec_conf, baseline_items, n_papers])
    # pd.DataFrame(data, columns=['Nt', 'J', 'lr', 'loss_mean', 'loss_std',
    #                             'price_mean', 'price_std', 'alg', 'recall', 'precision',
    #                             'f_beta', 'tests_num', 'corr', 'machine_selec_conf', 'baseline_items',
    #                             'total_items']).to_csv('output/data/fig6_machine_selec_conf.csv', index=False)
