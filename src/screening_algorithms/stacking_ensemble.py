from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn import grid_search
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

from src.screening_algorithms.machine_ensemble import MachineEnsemble


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
        self.lr = params['lr']
        self.machines_accuracy = params['machines_accuracy']
        self.ground_truth = params['ground_truth']
        self.ground_truth_tests = params['ground_truth_tests']
        self.machine_test_votes = params['machine_test_votes']
        self.votes_list = params['votes_list']

        # metrics to be computed
        self.loss = None
        self.recall = None
        self.precision = None
        self.f_beta = None
        self.price_per_paper = None

    def run(self):
        # create stacking classifier
        meta_clf = MetaClassifier(LogisticRegression(class_weight={1: 1, 0: 10}))
        meta_clf.fit(self.machine_test_votes, self.ground_truth_tests)

        # ensemble votes for each filter and item
        ensembled_votes = list(meta_clf.predict_proba(self.votes_list))

        items_labels = self._classify_items(ensembled_votes)
        metrics = self.compute_metrics(items_labels, self.ground_truth, self.lr, self.filters_num)
        self.loss = metrics[0]
        self.recall = metrics[1]
        self.precision = metrics[2]
        self.f_beta = metrics[3]

        return self.loss, self.recall, self.precision, self.f_beta, ensembled_votes
