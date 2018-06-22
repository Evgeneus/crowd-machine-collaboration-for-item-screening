from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn import grid_search
from sklearn.preprocessing import PolynomialFeatures

from src.screening_algorithms.machine_ensemble import MachineEnsemble

import numpy as np
from scipy.stats import beta
from random import shuffle


class MetaClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, clf, weights=None):
        self.clf = clf
        self.weights = weights
        # polynomial features
        self.poly_degree = 1
        self.poly = PolynomialFeatures(self.poly_degree, include_bias=False)

    def fit(self, test_votes, y):
        # stack predicted labels
        test_votes_stacked = np.dstack(test_votes)[0]
        X = test_votes_stacked

        # create polynom features
        # X = self._transform_features(test_votes_stacked)

        # parameters for greed search
        grid_values = {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

        # define classifier
        self.clf = grid_search.GridSearchCV(self.clf, grid_values, scoring='neg_log_loss')
        self.clf.fit(X, y)

    def predict(self, labels):
        X = self._transform_features(labels)

        return self.clf.predict(X)

    def predict_proba(self, labels):
        X = self._transform_features(labels)

        return self.clf.predict_proba(X)[:, 0]

    # create polynom features
    def _transform_features(self, labels):
        return self.poly.fit_transform(labels)


class StackingEnsemble(MachineEnsemble):

    def __init__(self, params):
        self.filters_num = params['filters_num']
        self.items_num = params['items_num']
        self.ground_truth = params['ground_truth']
        self.lr = params['lr']
        self.corr = params['corr']
        self.machine_tests = params['machine_tests']
        self.select_conf = params['select_conf']
        self.machines_num = params['machines_num']
        self.machines_passed_num = None

        self.machines_accuracy = params['machines_accuracy']
        # self.estimated_acc = params['estimated_acc']
        self.ground_truth_tests = params['ground_truth_tests']
        self.machine_test_votes = params['machine_test_votes']
        self.votes_list = params['votes_list']
        self.machines_passed_num = len(self.machines_accuracy)

        # metrics to be computed
        self.loss = None
        self.recall = None
        self.precision = None
        self.f_beta = None
        self.price_per_paper = None

    def run(self):
        # machines_accuracy = self._get_machines()

        # generate test data and ground truth values for the tests
        # test_votes, ground_truth_tests = self._generate_test_votes(machines_accuracy)

        # create stacking classifier
        meta_clf = MetaClassifier(LogisticRegression(class_weight={1: 1, 0: 10}))
        meta_clf.fit(self.machine_test_votes, self.ground_truth_tests)

        # votes_list = [[] for _ in range(self.items_num * self.filters_num)]

        # # generate votes for the first machine
        # first_machine_acc = machines_accuracy[0]
        # for item_index in range(self.items_num):
        #     for filter_index in range(self.filters_num):
        #         gt = self.ground_truth[item_index * self.filters_num + filter_index]  # can be either 0 or 1
        #         if np.random.binomial(1, first_machine_acc):
        #             vote = gt
        #         else:
        #             vote = 1 - gt
        #         votes_list[item_index * self.filters_num + filter_index].append(vote)
        #
        # # generate votes for the rest machines
        # rest_machine_acc = machines_accuracy[1:]
        # for item_index in range(self.items_num):
        #     for filter_index in range(self.filters_num):
        #         gt = self.ground_truth[item_index * self.filters_num + filter_index]  # can be either 0 or 1
        #         vote_prev = votes_list[item_index * self.filters_num + filter_index][-1]
        #         for machine_acc in rest_machine_acc:
        #             vote = self._generate_vote(gt, machine_acc, vote_prev)
        #             votes_list[item_index * self.filters_num + filter_index].append(vote)

        # ensemble votes for each filter and item
        ensembled_votes = list(meta_clf.predict_proba(self.votes_list))

        items_labels = self._classify_items(ensembled_votes)
        metrics = self.compute_metrics(items_labels, self.ground_truth, self.lr, self.filters_num)
        self.loss = metrics[0]
        self.recall = metrics[1]
        self.precision = metrics[2]
        self.f_beta = metrics[3]
        return self.loss, self.recall, self.precision, self.f_beta, ensembled_votes

    # def _get_machines(self):
    #     test_votes = [[] for _ in range(self.machines_num)]
    #
    #     # generate accuracy of machines
    #     machines_acc = np.random.uniform(0.5, 0.95, self.machines_num)
    #     first_machine_acc = machines_acc[0]
    #
    #     # set votes on tests that are generated by first machine
    #     test_votes[0] = list(np.random.binomial(1, first_machine_acc, self.machine_tests))
    #
    #     # generate votes for the rest machines to be tested
    #     for m_id, acc in enumerate(machines_acc[1:]):
    #         for i in range(self.machine_tests):
    #             if np.random.binomial(1, self.corr):
    #                 vote = test_votes[m_id][i]
    #             else:
    #                 vote = np.random.binomial(1, acc)
    #             test_votes[m_id + 1].append(vote)
    #
    #     selected_machines_acc = []
    #     # estimated_acc = []
    #     for machine_votes, acc in zip(test_votes, machines_acc):
    #         correct_votes_num = sum(machine_votes)
    #         conf = beta.sf(0.5, correct_votes_num + 1, self.machine_tests - correct_votes_num + 1)
    #         if conf > self.select_conf:
    #             selected_machines_acc.append(acc)
    #
    #     # check number of machines passed tests
    #     # add at least one machine passed tests (accuracy in [0.55, 0.9])
    #     if len(selected_machines_acc) < 1:
    #         m_acc = np.random.uniform(0.55, 0.9)
    #         selected_machines_acc.append(m_acc)
    #     self.machines_passed_num = len(selected_machines_acc)
    #
    #     return selected_machines_acc

    # def _generate_test_votes(self, machines_accuracy):
    #     test_votes = [[] for _ in range(self.machines_passed_num)]
    #     # ground_truth_tests = np.random.binomial(1, 0.5, self.machine_tests)
    #     ground_truth_tests = [0]*(self.machine_tests//2) + [1]*(self.machine_tests//2)
    #     shuffle(ground_truth_tests)
    #
    #     # set votes on tests that are generated by first machine
    #     first_machine_acc = machines_accuracy[0]
    #     while True:
    #         correct_votes_num = 0
    #         for gt in ground_truth_tests:
    #             if np.random.binomial(1, first_machine_acc):
    #                 test_votes[0].append(gt)
    #                 correct_votes_num += 1
    #             else:
    #                 test_votes[0].append(1 - gt)
    #         conf = beta.sf(0.5, correct_votes_num + 1, self.machine_tests - correct_votes_num + 1)
    #         if conf > self.select_conf:
    #             break
    #         else:
    #             test_votes[0] = []
    #
    #     # generate votes for the rest machines to be tested
    #     for m_id, acc in enumerate(machines_accuracy[1:]):
    #         while True:
    #             for i, gt in enumerate(ground_truth_tests):
    #                 prev_machine_vote = test_votes[m_id][i]
    #                 if np.random.binomial(1, self.corr):
    #                     if prev_machine_vote != gt:
    #                         vote = prev_machine_vote
    #                     else:
    #                         if np.random.binomial(1, acc):
    #                             vote = gt
    #                         else:
    #                             vote = 1 - gt
    #                 else:
    #                     if np.random.binomial(1, acc):
    #                         vote = gt
    #                     else:
    #                         vote = 1 - gt
    #                 test_votes[m_id + 1].append(vote)
    #
    #             # check the condition of selection confidence
    #             correct_votes_num = 0
    #             for i, j in zip(test_votes[m_id + 1], ground_truth_tests):
    #                 if i == j:
    #                     correct_votes_num += 1
    #
    #             conf = beta.sf(0.5, correct_votes_num + 1, self.machine_tests - correct_votes_num + 1)
    #             if conf > self.select_conf:
    #                 break
    #             else:
    #                 test_votes[m_id + 1] = []
    #
    #     return test_votes, ground_truth_tests
