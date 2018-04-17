import numpy as np
from scipy.stats import beta

from src.screening_algorithms.helpers.utils import Metrics


class MachineEnsemble(Metrics):

    def __init__(self, params):
        self.filters_num = params['filters_num']
        self.items_num = params['items_num']
        self.ground_truth = params['ground_truth']
        self.lr = params['lr']
        self.machines_num = params['machines_num']
        self.corr = [0.]*(self.machines_num//2) + [params['corr']]*(self.machines_num//2)
        self.machine_tests = params['machine_tests']
        self.select_conf = params['select_conf']
        # metrics to be computed
        self.loss = None
        self.recall = None
        self.precision = None
        self.f_beta = None
        self.price_per_paper = None

    def run(self):
        machines_accuracy, estimated_acc = self._get_machines()

        votes_list = [[] for _ in range(self.items_num * self.filters_num)]

        # generate votes for the first machine
        first_machine_acc = machines_accuracy[0]
        for item_index in range(self.items_num):
            for filter_index in range(self.filters_num):
                gt = self.ground_truth[item_index * self.filters_num + filter_index]  # can be either 0 or 1
                if np.random.binomial(1, first_machine_acc):
                    vote = gt
                else:
                    vote = 1 - gt
                votes_list[item_index * self.filters_num + filter_index].append(vote)

        # generate votes for the rest machines
        rest_machine_acc = machines_accuracy[1:]
        for item_index in range(self.items_num):
            for filter_index in range(self.filters_num):
                gt = self.ground_truth[item_index * self.filters_num + filter_index]  # can be either 0 or 1
                vote_prev = votes_list[item_index * self.filters_num + filter_index][-1]
                for m_id, machine_acc in enumerate(rest_machine_acc):
                    vote = self._generate_vote(gt, machine_acc, vote_prev, m_id)
                    votes_list[item_index * self.filters_num + filter_index].append(vote)

        # ensemble votes for each filter and item
        ensembled_votes = self._naive_bayes(votes_list, estimated_acc)

        items_labels = self._classify_items(ensembled_votes)
        metrics = self.compute_metrics(items_labels, self.ground_truth, self.lr, self.filters_num)
        self.loss = metrics[0]
        self.recall = metrics[1]
        self.precision = metrics[2]
        self.f_beta = metrics[3]
        return self.loss, self.recall, self.precision, self.f_beta, ensembled_votes, machines_accuracy

    def _get_machines(self):
        test_votes = [[] for _ in range(self.machines_num)]

        # generate accuracy of machines
        machines_acc = np.random.uniform(0.5, 0.95, self.machines_num)
        first_machine_acc = machines_acc[0]

        # set votes on tests that are generated by first machine
        test_votes[0] = list(np.random.binomial(1, first_machine_acc, self.machine_tests))

        # generate votes for the rest machines to be tested
        for m_id, acc in enumerate(machines_acc[1:]):
            m_corr = self.corr[m_id+1]
            for i in range(self.machine_tests):
                if np.random.binomial(1, m_corr):
                    vote = test_votes[m_id][i]
                else:
                    vote = np.random.binomial(1, acc)
                test_votes[m_id + 1].append(vote)

        selected_machines_acc = []
        estimated_acc = []
        for machine_votes, acc in zip(test_votes, machines_acc):
            correct_votes_num = sum(machine_votes)
            conf = beta.sf(0.5, correct_votes_num + 1, self.machine_tests - correct_votes_num + 1)
            if conf > self.select_conf:
                selected_machines_acc.append(acc)
                m_acc = correct_votes_num / self.machine_tests
                if m_acc > 0.95:
                    m_acc = 0.95
                estimated_acc.append(m_acc)

        # check number of machines passed tests
        # add at least one machine passed tests (accuracy in [0.55, 0.9])
        if len(selected_machines_acc) < 1:
            m_acc = np.random.uniform(0.55, 0.9)
            selected_machines_acc.append(m_acc)
            estimated_acc.append(m_acc)
        if len(selected_machines_acc) < 5:
            print('<5')
        return selected_machines_acc, estimated_acc

    def _generate_vote(self, gt, acc, vote_prev, m_id):
        m_corr = self.corr[m_id+1]
        if np.random.binomial(1, m_corr, 1)[0]:
            vote = vote_prev
        else:
            if np.random.binomial(1, acc):
                vote = gt
            else:
                vote = 1 - gt
        return vote

    # fuse votes via weighted majority voting
    # output_data: probabilities to be negatives for each filter and item
    def _naive_bayes(self, votes_list, estimated_acc):
        probs_list = [None] * self.filters_num * self.items_num
        for filter_index in range(self.filters_num):
            filter_machines_acc = estimated_acc
            for item_index in range(self.items_num):
                like_true_val = 1  # assume true value is positive
                a, b = 1., 1.  # constituents of baysian formula, prior is uniform dist.
                # a responds for positives, b - for negatives
                for vote, acc in zip(votes_list[item_index * self.filters_num + filter_index], filter_machines_acc):
                    if vote == like_true_val:
                        a *= acc
                        b *= 1 - acc
                    else:
                        a *= 1 - acc
                        b *= acc
                probs_list[item_index * self.filters_num + filter_index] = b / (a + b)
        return probs_list

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
