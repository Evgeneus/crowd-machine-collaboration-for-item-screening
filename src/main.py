import numpy as np
import pandas as pd

from src.screening_algorithms.helpers.utils import Generator
from src.screening_algorithms.helpers.utils import Workers
from src.screening_algorithms.machine_ensemble import MachineEnsemble
from src.screening_algorithms.s_run import SRun

'''
z - proportion of cheaters
lr - loss ration, i.e., how much a False Negative is more harmful than a False Positive
votes_per_item - crowd votes per item for base round
worker_tests - number of test questions per worker
machine_tests - number of test items per machine classifier
corr - correlation of errors between machine classifiers
expert_cost - cost of an expert to label a paper (i.e., labels on all filters)
select_conf - confidence level that a machine has accuracy > 0.5
theta - overall proportion of positive items
filters_num - number of filters
filters_select - selectivity of filters (probability of applying a filter)
filters_dif - difficulty of filters
iter_num - number of iterations for averaging results
'''


if __name__ == '__main__':
    z = 0.3
    items_num = 1000
    items_per_worker = 10
    baseround_items = 20  # must be a multiple of items_per_worker
    if baseround_items % items_per_worker:
        raise ValueError('baseround_items must be a multiple of items_per_worker')
    select_conf = 0.95
    worker_tests = 5
    votes_per_item = 3
    machine_tests = 50
    lr = 10
    expert_cost = 20
    filters_num = 4
    theta = 0.3
    filters_select = [0.14, 0.14, 0.28, 0.42]
    # filters_dif = [1., 1., 1.1, 0.9]  # old difficulty
    filters_dif = [0.9, 1., 1.1, 1.]
    iter_num = 50
    data = []

    # # ------------------------
    # # experiment on theta
    # for theta, x in zip([0.05, 0.1], [0.26, 0.23]):
    # for theta, x in zip([0.2, 0.4, 0.5], [0.18, 0.11, 0.09]):

    # # ------------------------
    # # experiment on a filter
    # filters_select_list = [[0.14, 0.14, 0.14, 0.56],  # High select
    #                        [0.14, 0.14, 0.14, 0.56],  # High select
    #                        [0.25, 0.25, 0.25, 0.098],  # Low select
    #                        [0.25, 0.25, 0.25, 0.098]]  # Low select
    # filters_dif_list = [[1., 1., 1., 1.2],  # High accurate
    #                     [1., 1., 1., 0.7],  # Low accurate
    #                     [1., 1., 1., 1.2],  # High accurate
    #                     [1., 1., 1., 0.7]]  # Low accurate
    # labels = ['High Select/High Acc', 'High Select/Low Acc', 'Low Select/High Acc', 'Low Select/Low Acc']
    # for filters_select, filters_dif, label in zip(filters_select_list, filters_dif_list, labels):
    #     print(label)

    # # --------------------
    # # experiment: number of filters
    # filters_select_list = [[0.14, 0.14, 0.14, 0.56],
    #                        [0.15, 0.15, 0.56],
    #                        [0.16, 0.16*4],
    #                        [0.3]]
    # filters_dif_list = [[1.]*4,
    #                    [1.]*3,
    #                    [1.]*2
    #                    [1]]
    # filters_num_list = [4, 3, 2, 1]
    # for filters_select, filters_dif, filters_num in zip(filters_select_list, filters_dif_list, labels):


    # # ------------------------
    # for theta in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
    # for filters_num in [1, 2, 3, 4, 5]:
    #     pow = 1 - theta**(1/filters_num)
    #     filters_select = [pow]*filters_num
    #     filters_dif = [1.]*filters_num
    # for expert_cost in [10, 20, 30, 40, 50, 70, 100]:
    params = {
        'filters_num': filters_num,
        'items_num': items_num,
        'baseround_items': baseround_items,
        'items_per_worker': items_per_worker,
        'votes_per_item': votes_per_item,
        'filters_select': filters_select,
        'filters_dif': filters_dif,
        'worker_tests': worker_tests,
        'lr': lr,
        'expert_cost': expert_cost,
        'stop_score': 30
    }

    # S-run algorithm
    loss_smrun_list = []
    cost_smrun_list = []
    rec_sm, pre_sm, f_sm, f_sm = [], [], [], []
    for _ in range(iter_num):
        # quiz, generation votes
        # workers = Workers(worker_tests, z)
        workers_accuracy = Workers(worker_tests, z).simulate_workers()
        params.update({'workers_accuracy': workers_accuracy,
                       'ground_truth': None
                       })

        _, ground_truth = Generator(params).generate_votes_gt(items_num)
        params.update({'ground_truth': ground_truth})

        # s-run
        loss_smrun, cost_smrun, rec_sm_, pre_sm_, f_beta_sm = SRun(params).run()
        loss_smrun_list.append(loss_smrun)
        cost_smrun_list.append(cost_smrun)
        rec_sm.append(rec_sm_)
        pre_sm.append(pre_sm_)
        f_sm.append(f_beta_sm)

    data.append([worker_tests, worker_tests, lr, np.mean(loss_smrun_list), np.std(loss_smrun_list),
                 np.mean(cost_smrun_list), np.std(cost_smrun_list), 'Crowd-Ensemble', np.mean(rec_sm),
                 np.std(rec_sm), np.mean(pre_sm), np.std(pre_sm), np.mean(f_sm), np.std(f_sm),
                 0., 0., select_conf, baseround_items, items_num, expert_cost, theta, filters_num])

    print('SM-RUN    loss: {:1.3f}, loss_std: {:1.3f}, recall: {:1.2f}, rec_std: {:1.3f}, '
          'price: {:1.2f}, price_std: {:1.2f}, precision: {:1.3f}, f_b: {}'
          .format(np.mean(loss_smrun_list), np.std(loss_smrun_list), np.mean(rec_sm),
                  np.std(rec_sm), np.mean(cost_smrun_list), np.std(cost_smrun_list),
                  np.mean(pre_sm), np.mean(f_sm)))

    for machine_tests in [50]:
    # for machine_tests in [15, 20, 30, 40, 50, 100, 150, 200, 500]:
    # for select_conf in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]:
        # Machine and Hybrid algorithms
        for corr in [0., 0.2, 0.3, 0.5, 0.7, 0.9]:
            print('Theta: {}, filters_num: {}, Corr: {}, test_num: {}, baseround_items: {}, lr: {},'
                  ' select_conf: {}, expert_vote_cost: {}'.
                  format(theta, filters_num, corr, machine_tests, baseround_items, lr, select_conf, expert_cost))
            loss_me_list = []
            rec_me, pre_me, f_me, f_me = [], [], [], []

            loss_h_list = []
            cost_h_list = []
            rec_h, pre_h, f_h, f_h = [], [], [], []

            for _ in range(iter_num):
                # quiz, generation votes
                workers_accuracy = Workers(worker_tests, z).simulate_workers()
                params.update({'workers_accuracy': workers_accuracy,
                               'ground_truth': None
                               })

                _, ground_truth = Generator(params).generate_votes_gt(items_num)
                params.update({'ground_truth': ground_truth})

                params.update({
                    'corr': corr,
                    'machine_tests': machine_tests,
                    'select_conf': select_conf,
                    'ground_truth': ground_truth,
                    'workers_accuracy': workers_accuracy,
                    'stop_score': 15
                })

                # machine ensemble
                loss_me, rec_me_, pre_me_, f_beta_me, prior_prob_pos = MachineEnsemble(params).run()
                loss_me_list.append(loss_me)
                rec_me.append(rec_me_)
                pre_me.append(pre_me_)
                f_me.append(f_beta_me)

                # s-run with machine prior
                params['prior_prob_pos'] = prior_prob_pos

                loss_h, cost_h, rec_h_, pre_h_, f_beta_h = SRun(params).run()
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

            data.append([worker_tests, worker_tests, lr, np.mean(loss_me_list), np.std(loss_me_list), 0.,
                         0., 'Machines-Ensemble', np.mean(rec_me), np.std(rec_me),
                         np.mean(pre_me), np.std(pre_me), np.mean(f_me), np.std(f_me), machine_tests, corr,
                         select_conf, baseround_items, items_num, expert_cost, theta, filters_num])

            data.append([worker_tests, worker_tests, lr, np.mean(loss_h_list), np.std(loss_h_list),
                         np.mean(cost_h_list), np.std(cost_h_list), 'Hybrid-Ensemble', np.mean(rec_h),
                         np.std(rec_h), np.mean(pre_h), np.std(pre_h), np.mean(f_h), np.std(f_h),
                         machine_tests, corr, select_conf, baseround_items, items_num, expert_cost,
                         theta, filters_num])

    pd.DataFrame(data,
                 columns=['worker_tests', 'worker_tests', 'lr', 'loss_mean', 'loss_std', 'price_mean', 'price_std',
                          'algorithm', 'recall', 'recall_std', 'precision', 'precision_std',
                          'f_beta', 'f_beta_std', 'machine_tests', 'corr', 'select_conf', 'baseround_items',
                          'total_items', 'expert_cost', 'theta', 'filters_num']
                 ).to_csv('../data/output_data/figXXX.csv', index=False)
