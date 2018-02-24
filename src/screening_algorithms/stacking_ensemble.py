from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation, metrics, grid_search, svm
# from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from src.screening_algorithms.machine_ensemble import MachineEnsemble

import pandas as pd
import numpy as np
from scipy.stats import beta
import random
import copy


class MetaClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, clf, weights=None):
        self.clf = clf
        self.weights = weights
        # polynomial features
        self.poly_degree = 2
        self.poly = PolynomialFeatures(self.poly_degree, include_bias=False)

    def fit(self, test_votes, y):
        # stack predicted labels
        test_votes_stacked = np.dstack(test_votes)[0]

        # create polynom features
        X = self._transform_features(test_votes_stacked)

        # parameters for greed search
        grid_values = {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

        # define classifier
        self.clf = grid_search.GridSearchCV(self.clf, grid_values)
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
        self.machines_num = 5
        # training/validation
        self.meta_clf = None
        self.test_votes = None
        self.ground_truth_tests = None
        self.train_pos_num = 0
        self.train_neg_num = 0
        self.machines_accuracy = None
        self.votes_list = [[] for _ in range(self.items_num * self.filters_num)]
        # metrics to be computed
        self.loss = None
        self.recall = None
        self.precision = None
        self.f_beta = None
        self.price_per_paper = None

    def run(self):
        self._generate_machines()

        # generate test data and ground truth values for the tests
        self._generate_test_votes()

        # create stacking classifier
        self.meta_clf = MetaClassifier(LogisticRegression(class_weight={1: 1, 0: 10}))
        self.meta_clf.fit(self.test_votes, self.ground_truth_tests)

        # votes_list = [[] for _ in range(self.items_num * self.filters_num)]

        # generate votes for the first machine
        first_machine_acc = self.machines_accuracy[0]
        for item_index in range(self.items_num):
            for filter_index in range(self.filters_num):
                gt = self.ground_truth[item_index * self.filters_num + filter_index]  # can be either 0 or 1
                if np.random.binomial(1, first_machine_acc):
                    vote = gt
                else:
                    vote = 1 - gt
                self.votes_list[item_index * self.filters_num + filter_index].append(vote)

        # generate votes for the rest machines
        rest_machine_acc = self.machines_accuracy[1:]
        for item_index in range(self.items_num):
            for filter_index in range(self.filters_num):
                gt = self.ground_truth[item_index * self.filters_num + filter_index]  # can be either 0 or 1
                vote_prev = self.votes_list[item_index * self.filters_num + filter_index][0]
                for machine_acc in rest_machine_acc:
                    vote = self._generate_vote(gt, machine_acc, vote_prev)
                    self.votes_list[item_index * self.filters_num + filter_index].append(vote)

        # ensemble votes for each filter and item
        ensembled_votes = list(self.meta_clf.predict_proba(self.votes_list))

        items_labels = self._classify_items(ensembled_votes)
        metrics = self.compute_metrics(items_labels, self.ground_truth, self.lr, self.filters_num)
        self.loss = metrics[0]
        self.recall = metrics[1]
        self.precision = metrics[2]
        self.f_beta = metrics[3]

        return self.loss, self.recall, self.precision, self.f_beta, ensembled_votes

    def _generate_machines(self):
        self.machines_accuracy = np.random.uniform(0.5, 0.95, self.machines_num)

    def _generate_test_votes(self):
        test_votes = [[] for _ in range(self.machines_num)]
        ground_truth_tests = np.random.binomial(1, 0.7, self.machine_tests)

        # set votes on tests that are generated by first machine
        first_machine_acc = self.machines_accuracy[0]
        while True:
            correct_votes_num = 0
            for gt in ground_truth_tests:
                if np.random.binomial(1, first_machine_acc):
                    test_votes[0].append(gt)
                    correct_votes_num += 1
                else:
                    test_votes[0].append(1 - gt)
            conf = beta.sf(0.5, correct_votes_num + 1, self.machine_tests - correct_votes_num + 1)
            if conf > self.select_conf:
                break
            else:
                test_votes[0] = []

        # generate votes for the rest machines to be tested
        for m_id, acc in enumerate(self.machines_accuracy[1:]):
            while True:
                for i, gt in enumerate(ground_truth_tests):
                    if np.random.binomial(1, self.corr):
                        vote = test_votes[m_id][i]
                    else:
                        if np.random.binomial(1, acc):
                            vote = gt
                        else:
                            vote = 1 - gt
                    test_votes[m_id + 1].append(vote)

                # check the condition of selection confidence
                correct_votes_num = 0
                for i, j in zip(test_votes[m_id + 1], ground_truth_tests):
                    if i == j:
                        correct_votes_num += 1

                conf = beta.sf(0.5, correct_votes_num + 1, self.machine_tests - correct_votes_num + 1)
                if conf > self.select_conf:
                    break
                else:
                    test_votes[m_id + 1] = []

        self.test_votes = test_votes
        self.ground_truth_tests = ground_truth_tests

    def retrain(self, train_pos_num, train_neg_num):
        X = list(copy.deepcopy(self.test_votes))
        y = list(copy.deepcopy(self.ground_truth_tests))

        self.train_pos_num += train_pos_num
        self.train_neg_num += train_neg_num
        y_active = [1]*train_pos_num + [0]*train_neg_num
        random.shuffle(y_active)
        y += y_active

        # generate votes for first machine
        first_machine_acc = self.machines_accuracy[0]
        first_machine_X = []
        for gt in y_active:
            if np.random.binomial(1, first_machine_acc):
                vote = gt
            else:
                vote = 1 - gt
            first_machine_X.append(vote)
        X[0] += first_machine_X

        # generate votes for the rest machines
        for m_id, clf_acc in enumerate(self.machines_accuracy[1:]):
            for i, gt in enumerate(y_active):
                if np.random.binomial(1, self.corr):
                    vote = X[m_id][i+self.machine_tests]
                else:
                    if np.random.binomial(1, clf_acc):
                        vote = gt
                    else:
                        vote = 1 - gt
                X[m_id+1].append(vote)

        # create stacking classifier
        self.meta_clf = MetaClassifier(LogisticRegression(class_weight={1: 1, 0: 10}))
        self.meta_clf.fit(X, y)
        # ensemble votes for each filter and item
        ensembled_votes = list(self.meta_clf.predict_proba(self.votes_list))
        return ensembled_votes
