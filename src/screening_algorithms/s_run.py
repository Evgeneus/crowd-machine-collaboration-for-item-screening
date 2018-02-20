from src.fusion_algorithms.em import expectation_maximization
from src.screening_algorithms.helpers.utils import Metrics
from src.fusion_algorithms.algorithms_utils import input_adapter
from src.screening_algorithms.helpers.s_run_utils import SRunUtils
from src.screening_algorithms.helpers.utils import Generator


class SRun(Generator, SRunUtils, Metrics):

    def __init__(self, params):
        self.filters_num = params['filters_num']
        self.items_num = params['items_num']
        self.items_per_worker = params['items_per_worker']
        self.votes_per_item = params['votes_per_item']
        self.lr = params['lr']
        self.worker_tests = params['worker_tests']
        self.workers_accuracy = params['workers_accuracy']
        self.filters_select = params['filters_select']
        self.filters_dif = params['filters_dif']
        self.ground_truth = params['ground_truth']
        self.baseround_items = params['baseround_items']
        self.expert_cost = params['expert_cost']
        self.stop_score = params['stop_score']
        self.prior_prob_pos = params.get('prior_prob_pos')
        self.p_thrs = 0.99

        # measurements to be computed
        self.filters_list = list(range(self.filters_num))
        self.votes_count = 0  # budget spent
        self.filters_select_est = []
        self.filters_acc_est = []
        self.votes_stats = [[0, 0] for _ in range(self.items_num * self.filters_num)]
        self.items_classified = dict(zip(range(self.items_num), [1] * self.items_num))
        # metrics to be computed
        self.loss = None
        self.recall = None
        self.precision = None
        self.f_beta = None
        self.price_per_paper = None

    def run(self):
        # base round
        self.votes_count += (self.worker_tests + self.items_per_worker * self.filters_num)\
                            * self.votes_per_item * self.baseround_items // self.items_per_worker

        items_classified_baseround, items_to_classify = self._do_baseround()

        # check for bordercases
        for filter_index, filter_acc in enumerate(self.filters_acc_est):
            if filter_acc > 0.98:
                self.filters_acc_est[filter_index] = 0.95

        self.items_classified.update(items_classified_baseround)
        items_to_classify = items_to_classify + list(range(self.baseround_items, self.items_num))

        # compute prior power
        if self.prior_prob_pos:
            self.filters_select_est = [0.] * self.filters_num
            for item_index in range(self.items_num):
                for filter_index in range(self.filters_num):
                    self.filters_select_est[filter_index] += 1 - self.prior_prob_pos[item_index
                                                               * self.filters_num + filter_index]
            self.filters_select_est = list(map(lambda x: x / self.items_num, self.filters_select_est))

        # Do Multi rounds
        while len(items_to_classify) != 0:
            self.votes_count += len(items_to_classify)
            filters_assigned, items_to_classify = self.assign_filters(items_to_classify)

            votes = self._do_round(items_to_classify, filters_assigned)
            # update votes_stats
            self.update_votes_stats(filters_assigned, votes, items_to_classify)

            # update filters selectivity
            self.update_filters_select()

            # classify items
            items_classified_round, items_to_classify = self.classify_items(items_to_classify)
            self.items_classified.update(items_classified_round)

        self.items_classified = [self.items_classified[item_index] for item_index in sorted(self.items_classified.keys())]
        metrics = self.compute_metrics(self.items_classified, self.ground_truth, self.lr, self.filters_num)
        self.loss = metrics[0]
        self.recall = metrics[1]
        self.precision = metrics[2]
        self.f_beta = metrics[3]
        fp = metrics[4]
        self.price_per_paper = (self.votes_count + fp * self.expert_cost) / self.items_num

        return self.loss, self.price_per_paper, self.recall, self.precision, self.f_beta

    def _do_baseround(self):
        # generate votes
        votes = self.generate_votes_gt(self.baseround_items)
        # aggregate votes via truth finder
        psi = input_adapter(votes)
        n = (self.baseround_items // self.items_per_worker) * self.votes_per_item
        _, p = expectation_maximization(n, self.baseround_items * self.filters_num, psi)
        values_prob = []
        for e in p:
            e_prob = [0., 0.]
            for e_id, e_p in e.items():
                e_prob[e_id] = e_p
            values_prob.append(e_prob)

        self.estimate_filters_property(votes, self.baseround_items)
        items_classified, items_to_classify = self.classify_items_baseround(values_prob)
        # count value counts
        for key in range(self.baseround_items * self.filters_num):
            filter_item_votes = votes[key]
            for v in filter_item_votes.values():
                self.votes_stats[key][v[0]] += 1

        return items_classified, items_to_classify

    def _do_round(self, items, filters_assigned):
        n = len(items)
        items_per_worker = self.items_per_worker * self.filters_num
        items_batch1 = items[:n - n % items_per_worker]
        items_batch2 = items[n - n % items_per_worker:]

        votes_batch1 = self.generate_votes(filters_assigned, items_batch1)
        votes_batch2 = self.generate_votes(filters_assigned, items_batch2)

        votes = votes_batch1 + votes_batch2
        return votes
