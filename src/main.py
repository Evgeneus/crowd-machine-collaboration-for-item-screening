import numpy as np
import pandas as pd


from src.screening_algorithms.helpers.utils import Generator
from src.screening_algorithms.helpers.utils import Workers
from src.screening_algorithms.helpers.utils import Metrics
from src.screening_algorithms.machine_ensemble import MachineEnsemble
from src.screening_algorithms.s_run import SRun
from src.screening_algorithms.stacking_ensemble import StackingEnsemble

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


class BestMachine(Metrics):

    def __init__(self, estimated_acc, votes, params):
        self.estimated_acc = estimated_acc
        self.votes = votes
        self.filters_num = params['filters_num']
        self.items_num = params['items_num']
        self.ground_truth = params['ground_truth']
        self.lr = params['lr']

    def _classify_items(self, ensembled_votes):
        items_labels = []
        neg_thr = 0.99  # threshold to classify as a positive
        for item_index in range(self.items_num):
            prob_filters_not_apply = 1.
            for filter_index in range(self.filters_num):
                prob_filters_not_apply *= ensembled_votes[item_index * self.filters_num + filter_index]
            prob_item_out = 1. - prob_filters_not_apply

            # classify item
            if prob_item_out > neg_thr:
                items_labels.append(0)
            else:
                items_labels.append(1)
        return items_labels

    def get_metrics(self):
        probs = []
        for i in self.votes:
            probs.append(self.estimated_acc if i == 0 else 1 - self.estimated_acc)
        items_labels = self._classify_items(probs)
        metrics = self.compute_metrics(items_labels, self.ground_truth, self.lr, self.filters_num)
        self.loss = metrics[0]
        self.recall = metrics[1]
        self.precision = metrics[2]

        return self.loss, self.recall, self.precision


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
    machine_tests = 100
    machines_num = 10
    machine_acc_range = [0.5, 0.8]
    lr = 10
    expert_cost = 20
    filters_num = 4
    theta = 0.3
    filters_select = [0.14, 0.14, 0.28, 0.42]
    filters_dif = [0.9, 1., 1.1, 1.]
    iter_num = 50
    data = []

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
        workers_accuracy = Workers(worker_tests, z).simulate_workers()
        params.update({'workers_accuracy': workers_accuracy,
                       'ground_truth': None
                       })

        _, ground_truth = Generator(params).generate_votes_gt(items_num)
        params.update({'ground_truth': ground_truth})
    #
    #     # s-run
    #     loss_smrun, cost_smrun, rec_sm_, pre_sm_, f_beta_sm = SRun(params).run()
    #     loss_smrun_list.append(loss_smrun)
    #     cost_smrun_list.append(cost_smrun)
    #     rec_sm.append(rec_sm_)
    #     pre_sm.append(pre_sm_)
    #     f_sm.append(f_beta_sm)
    #
    # data.append([worker_tests, worker_tests, lr, np.mean(loss_smrun_list), np.std(loss_smrun_list),
    #              np.mean(cost_smrun_list), np.std(cost_smrun_list), 'Crowd-Ensemble', np.mean(rec_sm),
    #              np.std(rec_sm), np.mean(pre_sm), np.std(pre_sm), np.mean(f_sm), np.std(f_sm),
    #              0., 0., select_conf, baseround_items, items_num, expert_cost, theta, filters_num, None])
    #
    # print('SM-RUN    loss: {:1.3f}, loss_std: {:1.3f}, recall: {:1.2f}, rec_std: {:1.3f}, '
    #       'price: {:1.2f}, price_std: {:1.2f}, precision: {:1.3f}, f_b: {}'
    #       .format(np.mean(loss_smrun_list), np.std(loss_smrun_list), np.mean(rec_sm),
    #               np.std(rec_sm), np.mean(cost_smrun_list), np.std(cost_smrun_list),
    #               np.mean(pre_sm), np.mean(f_sm)))
    print('Theta: {}, filters_num: {}, test_num: {}, baseround_items: {}, lr: {},'
          ' select_conf: {}, expert_vote_cost: {}'.
          format(theta, filters_num, machine_tests, baseround_items, lr, select_conf, expert_cost))

    # Machine and Hybrid algorithms
    for corr in [0., 0.2, 0.3, 0.5, 0.7, 0.9]:
        print('Corr: {}'.format(corr))
        loss_me_list = []
        rec_me, pre_me, f_me, f_me = [], [], [], []

        loss_h_list = []
        cost_h_list = []
        rec_h, pre_h, f_h, f_h = [], [], [], []

        loss_hs_list = []
        cost_hs_list = []
        rec_hs, pre_hs, f_hs, f_hs = [], [], [], []

        loss_b_list, rec_b_list, pre_b_list, acc_b_clf = [], [], [], []

        delta_loss, delta_pre, delta_rec = [], [], []

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
                'machines_num': machines_num,
                'select_conf': select_conf,
                'ground_truth': ground_truth,
                'workers_accuracy': workers_accuracy,
                'machine_acc_range': machine_acc_range,
                'stop_score': 15
            })

            # machine ensemble
            loss_me, rec_me_, pre_me_, f_beta_me, prior_prob_pos, payload_list = MachineEnsemble(params).run()
            machines_accuracy, estimated_acc, ground_truth_tests, machine_test_votes, votes_list = payload_list
            loss_me_list.append(loss_me)
            rec_me.append(rec_me_)
            pre_me.append(pre_me_)
            f_me.append(f_beta_me)

            # best_machine
            b_loss, b_rec, b_pre = BestMachine(estimated_acc[0], [i[0] for i in votes_list], params).\
                get_metrics()
            loss_b_list.append(b_loss)
            pre_b_list.append(b_pre)
            rec_b_list.append(b_rec)
            acc_b_clf.append(estimated_acc[0])


            # # s-run with machine prior
            # params['prior_prob_pos'] = prior_prob_pos
            #
            # loss_h, cost_h, rec_h_, pre_h_, f_beta_h = SRun(params).run()
            # loss_h_list.append(loss_h)
            # cost_h_list.append(cost_h)
            # rec_h.append(rec_h_)
            # pre_h.append(pre_h_)
            # f_h.append(f_beta_h)

            # s-run with machine prior (stacking)
            params['machines_accuracy'] = machines_accuracy
            params['estimated_acc'] = estimated_acc
            params['ground_truth_tests'] = ground_truth_tests
            params['machine_test_votes'] = machine_test_votes
            params['votes_list'] = votes_list
            # params['prior_prob_pos'] = StackingEnsemble(params).run()[4]

            loss_hs, rec_hs_, pre_hs_, _, _ = StackingEnsemble(params).run()
            loss_hs_list.append(loss_hs)
            # cost_hs_list.append(cost_hs)
            rec_hs.append(rec_hs_)
            pre_hs.append(pre_hs_)
            # f_hs.append(f_beta_hs)

        delta_loss = np.array(loss_hs_list) - np.array(loss_me_list)
        delta_rec = np.array(rec_hs) - np.array(rec_me)
        delta_pre = np.array(pre_hs) - np.array(pre_me)

        # print results
        print('NB        loss: {:1.3f}, loss_std: {:1.3f}, recall: {:1.2f}, rec_std: {:1.3f}, '
              'precision: {:1.3f}, precision_std: {}'
              .format(np.mean(loss_me_list), np.std(loss_me_list), np.mean(rec_me),
                      np.std(rec_me), np.mean(pre_me), np.std(pre_me)))

        print('Reg       loss: {:1.3f}, loss_std: {:1.3f}, ' 'recall: {:1.2f}, rec_std: {:1.3f}, '
                'precision: {:1.3f}, precision_std: {}'
              .format(np.mean(loss_hs_list), np.std(loss_hs_list), np.mean(rec_hs), np.std(rec_hs),
                      np.mean(pre_hs), np.std(pre_hs)))
        print('BestM     loss: {:1.3f}, loss_std: {:1.3f}, ' 'recall: {:1.2f}, rec_std: {:1.3f}, '
              'precision: {:1.3f}, precision_std: {}, acc_of_clf: {:1.3f}'
              .format(np.mean(loss_b_list), np.std(loss_b_list), np.mean(rec_b_list), np.std(rec_b_list),
                      np.mean(pre_b_list), np.std(pre_b_list), np.mean(acc_b_clf)))

        print('delta(REG-NB)   loss: {:1.3f}, loss_std: {:1.3f}, ' 'recall: {:1.2f}, rec_std: {:1.3f}, '
                'precision: {:1.3f}, precision_std: {}'.format(np.mean(delta_loss), np.std(delta_loss),
                                                               np.mean(delta_rec), np.std(delta_rec),
                      np.mean(delta_pre), np.std(delta_pre)))

        # print('HS-RUN    loss: {:1.3f}, loss_std: {:1.3f}, ' 'recall: {:1.2f}, rec_std: {:1.3f}, '
        #       'price: {:1.2f}, price_std: {:1.2f}, precision: {:1.3f}, f_b: {}'
        #       .format(np.mean(loss_hs_list), np.std(loss_hs_list), np.mean(rec_hs), np.std(rec_hs),
        #               np.mean(cost_hs_list), np.std(cost_hs_list), np.mean(pre_hs), np.mean(f_hs)))
        print('---------------------')

    #     data.append([worker_tests, worker_tests, lr, np.mean(loss_me_list), np.std(loss_me_list), 0.,
    #                  0., 'Machines-Ensemble', np.mean(rec_me), np.std(rec_me),
    #                  np.mean(pre_me), np.std(pre_me), np.mean(f_me), np.std(f_me), machine_tests, corr,
    #                  select_conf, baseround_items, items_num, expert_cost, theta, filters_num, machine_acc_range])
    #
    #     data.append([worker_tests, worker_tests, lr, np.mean(loss_h_list), np.std(loss_h_list),
    #                  np.mean(cost_h_list), np.std(cost_h_list), 'Hybrid-Ensemble', np.mean(rec_h),
    #                  np.std(rec_h), np.mean(pre_h), np.std(pre_h), np.mean(f_h), np.std(f_h),
    #                  machine_tests, corr, select_conf, baseround_items, items_num, expert_cost,
    #                  theta, filters_num, machine_acc_range])
    #
    #     data.append([worker_tests, worker_tests, lr, np.mean(loss_hs_list), np.std(loss_hs_list),
    #                  np.mean(cost_hs_list), np.std(cost_hs_list), 'Hybrid-Ensemble (Regression)', np.mean(rec_hs),
    #                  np.std(rec_h), np.mean(pre_h), np.std(pre_h), np.mean(f_h), np.std(f_h),
    #                  machine_tests, corr, select_conf, baseround_items, items_num, expert_cost,
    #                  theta, filters_num, machine_acc_range])
    #
    # pd.DataFrame(data,
    #              columns=['worker_tests', 'worker_tests', 'lr', 'loss_mean', 'loss_std', 'price_mean', 'price_std',
    #                       'algorithm', 'recall', 'recall_std', 'precision', 'precision_std',
    #                       'f_beta', 'f_beta_std', 'machine_tests', 'corr', 'select_conf', 'baseround_items',
    #                       'total_items', 'expert_cost', 'theta', 'filters_num', 'machine_acc_range']
    #              ).to_csv('../data/output_data/new/figXXX.csv', index=False)



