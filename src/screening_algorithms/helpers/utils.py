import numpy as np


class Workers:

    def __init__(self, worker_tests, cheaters_prop):
        self.worker_tests = worker_tests
        self.cheaters_prop = cheaters_prop
        self.acc_passed_neg = []
        self.acc_passed_pos = []

    # simulate workers that pass a set of test questions
    def simulate_workers(self):
        for _ in range(100000):
            self._simulate_quiz()

        return [self.acc_passed_neg, self.acc_passed_pos]

    def _simulate_quiz(self):
        # decide if a worker a cheater
        if np.random.binomial(1, self.cheaters_prop):
            # worker_type is 'rand_ch'
            worker_acc_neg, worker_acc_pos = 0.5, 0.5
        else:
            # worker_type is 'worker'
            worker_acc_pos = 0.5 + (np.random.beta(1, 1) * 0.5)
            worker_acc_neg = worker_acc_pos + 0.1 if worker_acc_pos + 0.1 <= 1. else 1.

        # iterate over test questions
        for item_index in range(self.worker_tests):
            # decide if the test item is positive or negative (50+/50-)
            if np.random.binomial(1, 0.5):
                # if worker is mistaken exclude him
                if not np.random.binomial(1, worker_acc_pos):
                    return
            else:
                if not np.random.binomial(1, worker_acc_neg):
                    return
        self.acc_passed_pos.append(worker_acc_pos)
        self.acc_passed_neg.append(worker_acc_neg)


# output_data generator
def generate_gold_data(items_num, filters_select):
    gold_data = []
    for item_index in range(items_num):
        for filter_select in filters_select:
            if np.random.binomial(1, filter_select):
                val = 1
            else:
                val = 0
            gold_data.append(val)
    return gold_data


def generate_votes_gt(items_num, filters_select, items_per_worker, worker_tests, workers_accuracy, filters_dif, gt=None):
    if not gt:
        gt = generate_gold_data(items_num, filters_select)
        is_gt_genereated = True
    else:
        is_gt_genereated = False
    workers_accuracy_neg, workers_accuracy_pos = workers_accuracy

    # generate votes
    # on a page a worker see items_per_worker tasks (crowdflower style)
    pages_num = items_num // items_per_worker
    filters_num = len(filters_select)
    votes = {}
    for item_filter_index in range(pages_num*items_per_worker*filters_num):
        votes[item_filter_index] = {}
    for page_index in range(pages_num):
        for i in range(worker_tests):
            worker_id = page_index * worker_tests + i
            w_acc_pos = workers_accuracy_pos.pop()
            workers_accuracy[1].insert(0, w_acc_pos)
            w_acc_neg = workers_accuracy_neg.pop()
            workers_accuracy[0].insert(0, w_acc_neg)
            for item_index in range(page_index*items_per_worker, page_index*items_per_worker + items_per_worker, 1):
                filter_item_indices = range(item_index*filters_num, item_index*filters_num + filters_num, 1)
                is_item_pos = sum([gt[i] for i in filter_item_indices]) == 0
                if is_item_pos:
                    worker_acc = w_acc_pos
                else:
                    worker_acc = w_acc_neg
                for item_filter_index, f_diff in zip(filter_item_indices, filters_dif):
                    if np.random.binomial(1, worker_acc*f_diff if worker_acc*f_diff <= 1. else 1.):
                        vote = gt[item_filter_index]
                    else:
                        vote = 1 - gt[item_filter_index]
                    votes[item_filter_index][worker_id] = [vote]
    if is_gt_genereated:
        return votes, gt
    else:
        return votes


def compute_metrics(items, gt, lr, filters_num):
    # obtain ground_truth scope values for items
    gt_scope = []
    for item_index in range(len(items)):
        if sum([gt[item_index*filters_num + filter_index] for filter_index in range(filters_num)]):
            gt_scope.append(0)
        else:
            gt_scope.append(1)
    # Positive == Inclusion (Relevant)
    # Negative == Exclusion (Not relevant)
    fp = 0.
    fn = 0.
    tp = 0.
    tn = 0.
    for cl_val, gt_val in zip(items, gt_scope):
        if gt_val and not cl_val:
            fn += 1
        if not gt_val and cl_val:
            fp += 1
        if gt_val and cl_val:
            tp += 1
        if not gt_val and not cl_val:
            tn += 1
    recall = 100 * tp / (tp + fn)
    precision = 100 * tp / (tp + fp)
    loss = (fn * lr + fp) / len(items)
    beta = 1. / lr
    f_beta = (beta + 1) * precision * recall / (beta * recall + precision)
    return loss, recall, precision, f_beta, fp

