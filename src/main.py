from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn import grid_search
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import log_loss
import numpy as np
from src.screening_algorithms.machine_ensemble import MachineEnsemble


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


class MetaClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, clf, weights=None):
        self.clf = clf
        self.weights = weights
        # polynomial features
        self.poly_degree = 1
        self.poly = PolynomialFeatures(self.poly_degree, include_bias=False)

    def fit(self, X, y):
        # parameters for greed search
        grid_values = {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

        # define classifier
        self.clf = grid_search.GridSearchCV(self.clf, grid_values, scoring='neg_log_loss')
        self.clf.fit(X, y)

    def predict(self, labels):
        X = self._transform_features(labels)

        return self.clf.predict(X)

    def predict_proba(self, X):

        return self.clf.predict_proba(X)

    # create polynom features
    def _transform_features(self, labels):
        return self.poly.fit_transform(labels)


class NB:

    def __init__(self, filters_num, items_num, estimated_acc):
        self.filters_num = filters_num
        self.items_num = items_num
        self.estimated_acc = estimated_acc

    # output_data: probabilities to be negatives for each filter and item
    def predict_proba(self, votes_list):
        probs_list = [None] * self.filters_num * self.items_num
        for filter_index in range(self.filters_num):
            filter_machines_acc = self.estimated_acc
            for item_index in range(self.items_num):
                like_true_val = 1  # assume true value is positive
                a, b = 1., 1.  # constituents of bayesian formula, prior is uniform dist.
                # a responds for positives, b - for negatives
                for vote, acc in zip(votes_list[item_index * self.filters_num + filter_index], filter_machines_acc):
                    if vote == like_true_val:
                        a *= acc
                        b *= 1 - acc
                    else:
                        a *= 1 - acc
                        b *= acc
                probs_list[item_index * self.filters_num + filter_index] = [b / (a + b), a / (a + b)]
        return probs_list


if __name__ == '__main__':
    items_num = 1000
    select_conf = 0.95
    machine_tests = 500
    machines_num = 10
    machine_acc_range = [0.5, 0.8]
    # lr = 10
    # expert_cost = 20
    filters_num = 4
    theta = 0.3
    filters_select = [0.14, 0.14, 0.28, 0.42]
    # filters_dif = [0.9, 1., 1.1, 1.]
    iter_num = 50
    data = []

    # Machine and Hybrid algorithms
    log_loss_stat = {}
    for corr in [0., 0.2, 0.3, 0.5, 0.7, 0.9]:
        print('Corr: {}'.format(corr), 'test_machines_num: {}'.format(machines_num))

        nb_loss = []
        reg_loss = []
        for _ in range(iter_num):
            # generate Y test data
            Y_test = []
            for item_index in range(items_num):
                for filter_select in filters_select:
                    if np.random.binomial(1, filter_select):
                        val = 1
                    else:
                        val = 0
                    Y_test.append(val)

            params = {
                'corr': corr,
                'machine_tests': machine_tests,
                'machines_num': machines_num,
                'select_conf': select_conf,
                'ground_truth': Y_test,
                'machine_acc_range': machine_acc_range,
                'filters_select': filters_select,
                'filters_num': filters_num,
                'items_num': items_num
            }

            # generate data
            machines_accuracy, estimated_acc, Y_train, X_train, X_test = MachineEnsemble(params).run()


            # LOGISTICK REGRESSION
            logistic_regression = MetaClassifier(LogisticRegression(class_weight={1: 1, 0: 10}))
            logistic_regression.fit(X_train, Y_train)

            # ensemble votes for each filter and item
            predicted_prob_regression = list(logistic_regression.predict_proba(np.array(X_test)))
            log_loss_regression = log_loss(Y_test, predicted_prob_regression)
            reg_loss.append(log_loss_regression)
            # print(log_loss_regression)


            # NAIVE BAYESIAN
            predicted_prob_nb = NB(filters_num, items_num, estimated_acc).predict_proba(X_test)
            log_loss_nb = log_loss(Y_test, predicted_prob_nb)
            nb_loss.append(log_loss_nb)
            # print(log_loss_nb)

        log_loss_stat[corr] = {'reg': (np.mean(reg_loss), np.std(reg_loss)),
                               'nb': (np.mean(nb_loss), np.std(nb_loss))}

    print(log_loss_stat)
