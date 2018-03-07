import numpy as np
from scipy.special import binom
from src.fusion_algorithms.algorithms_utils import input_adapter
from src.fusion_algorithms.em import expectation_maximization


class SRunUtils:

    def assign_filters(self, items):
        filters_assigned = []
        items_new = []
        for item_index in items:
            classify_score = []
            n_min_list = []
            joint_prob_votes_neg = [1.] * self.filters_num
            for filter_index in self.filters_list:
                filter_acc = self.filters_acc_est[filter_index]
                filter_select = self.filters_select_est[filter_index]
                if self.prior_prob_pos:
                    prob_item_neg = 1 - self.prior_prob_pos[item_index * self.filters_num + filter_index]
                else:
                    prob_item_neg = self.filters_select_est[filter_index]
                pos_c, neg_c = self.votes_stats[item_index * self.filters_num + filter_index]
                for n in range(1, 11):
                    # new value is negative
                    prob_vote_neg = filter_acc * prob_item_neg + (1 - filter_acc) * (1 - prob_item_neg)
                    joint_prob_votes_neg[filter_index] *= prob_vote_neg
                    term_neg = binom(pos_c + neg_c + n, neg_c + n) * filter_acc ** (neg_c + n) \
                               * (1 - filter_acc) ** pos_c * filter_select
                    term_pos = binom(pos_c + neg_c + n, pos_c) * filter_acc ** pos_c \
                               * (1 - filter_acc) ** (neg_c + n) * (1 - filter_select)
                    prob_item_pos = term_pos * prob_vote_neg / (term_neg + term_pos)
                    prob_item_neg = 1 - prob_item_pos
                    if prob_item_neg >= 0.99:
                        classify_score.append(joint_prob_votes_neg[filter_index] / n)
                        n_min_list.append(n)
                        break
                    elif n == 10:
                        classify_score.append(joint_prob_votes_neg[filter_index] / n)
                        n_min_list.append(n)

            filter_ = classify_score.index(max(classify_score))
            n_min = n_min_list[filter_]
            joint_prob = joint_prob_votes_neg[filter_]

            if n_min / joint_prob < self.stop_score:
                filters_assigned.append(filter_)
                items_new.append(item_index)

        return filters_assigned, items_new

    def classify_items_baseround(self, values_prob):
        items_classified = {}
        items_to_classify = []
        for item_index in range(self.baseround_items):
            prob_pos = 1.
            for filter_index in self.filters_list:
                prob_pos *= values_prob[item_index * self.filters_num + filter_index][0]
            prob_neg = 1 - prob_pos

            if prob_neg > self.p_thrs:
                items_classified[item_index] = 0
            elif prob_pos > self.p_thrs:
                items_classified[item_index] = 1
            else:
                items_to_classify.append(item_index)

        return items_classified, items_to_classify

    def classify_items(self, items):
        items_classified = {}
        items_to_classify = []

        for item_index in items:
            prob_item_pos = 1.
            for filter_index in self.filters_list:
                filter_acc = self.filters_acc_est[filter_index]
                filter_select = self.filters_select_est[filter_index]

                pos_c, neg_c = self.votes_stats[item_index * self.filters_num + filter_index]
                if pos_c == 0 and neg_c == 0:
                    prob_filter_pos = 1 - filter_select
                else:
                    term_pos = binom(pos_c + neg_c, pos_c) * filter_acc ** pos_c \
                               * (1 - filter_acc) ** neg_c * (1 - filter_select)
                    term_neg = binom(pos_c + neg_c, neg_c) * filter_acc ** neg_c \
                               * (1 - filter_acc) ** pos_c * filter_select
                    prob_filter_pos = term_pos / (term_pos + term_neg)
                prob_item_pos *= prob_filter_pos
            prob_item_neg = 1 - prob_item_pos

            if prob_item_neg > self.p_thrs:
                items_classified[item_index] = 0
            elif prob_item_pos > self.p_thrs:
                items_classified[item_index] = 1
            else:
                items_to_classify.append(item_index)

        return items_classified, items_to_classify

    def generate_votes(self, filters_assigned, items):
        votes = []
        workers_num = 1 if self.items_num < self.items_per_worker else self.items_num // self.items_per_worker
        for worker_index in range(workers_num):
            # get worker's accuracy
            worker_acc_pos = self.workers_accuracy[1].pop()
            self.workers_accuracy[1].insert(0, worker_acc_pos)
            worker_acc_neg = self.workers_accuracy[0].pop()
            self.workers_accuracy[0].insert(0, worker_acc_neg)

            filter_item_pair = zip(filters_assigned[worker_index*self.items_per_worker:
                                   worker_index*self.items_per_worker + self.items_per_worker],
                                   items[worker_index*self.items_per_worker:worker_index
                                   * self.items_per_worker + self.items_per_worker])
            for filter_index, item_index in filter_item_pair:
                # update the worker's accuracy on the current item
                is_item_pos = sum(self.ground_truth[item_index*self.filters_num:
                              item_index*self.filters_num + self.filters_num]) == 0
                if is_item_pos:
                    worker_acc = worker_acc_pos
                else:
                    worker_acc = worker_acc_neg

                # generate vote
                value_gt = self.ground_truth[item_index*self.filters_num + filter_index]
                cr_dif = self.filters_dif[filter_index]
                if np.random.binomial(1, worker_acc*cr_dif if worker_acc*cr_dif <= 1. else 1.):
                    vote = value_gt
                else:
                    vote = 1 - value_gt
                votes.append(vote)

        return votes

    def update_votes_stats(self, filters_assigned, votes, items):
        for filter_index, vote, item_index in zip(filters_assigned, votes, items):
            if vote:
                self.votes_stats[item_index*self.filters_num + filter_index][1] += 1
            else:
                self.votes_stats[item_index*self.filters_num + filter_index][0] += 1

    def update_filters_select(self):
        apply_filters_prob = [[] for _ in self.filters_list]
        for item_index in range(self.items_num):
            for filter_index in self.filters_list:
                filter_acc = self.filters_acc_est[filter_index]
                filter_select = self.filters_select_est[filter_index]
                pos_c, neg_c = self.votes_stats[item_index*self.filters_num + filter_index]

                term_pos = binom(pos_c + neg_c, pos_c) * filter_acc ** pos_c \
                           * (1 - filter_acc) ** neg_c * (1 - filter_select)
                term_neg = binom(pos_c + neg_c, neg_c) * filter_acc ** neg_c \
                           * (1 - filter_acc) ** pos_c * filter_select
                prob_filter_neg = term_neg / (term_pos + term_neg)
                apply_filters_prob[filter_index].append(prob_filter_neg)
        self.filters_select_est = [np.mean(i) for i in apply_filters_prob]

    def estimate_filters_property(self, votes, items_num):
        psi = input_adapter(votes)
        n = (self.baseround_items // self.items_per_worker) * self.votes_per_item
        for filter_index in self.filters_list:
            item_filter_votes = psi[filter_index::self.filters_num]
            filter_acc_list, filter_select_list = expectation_maximization(n, items_num, item_filter_votes)
            filter_acc = np.mean(filter_acc_list)
            filter_select = 0.
            for i in filter_select_list:
                i_prob = [0., 0.]
                for i_index, i_p in i.items():
                    i_prob[i_index] = i_p
                filter_select += i_prob[1]
            filter_select /= self.baseround_items
            self.filters_select_est.append(filter_select)
            self.filters_acc_est.append(filter_acc)

# # a low accurate criteria
# def generate_responses(GT, papers_ids, criteria_num, papers_worker, acc, criteria_difficulty, cr_assigned):
#     responses = []
#     n = len(papers_ids)
#     workers_n = 1 if n < papers_worker else n // papers_worker
#     for w_ind in range(workers_n):
#         worker_acc_in = acc[1].pop()
#         acc[1].insert(0, worker_acc_in)
#         worker_acc_out = acc[0].pop()
#         acc[0].insert(0, worker_acc_out)
#         for cr, p_id in zip(cr_assigned[w_ind*papers_worker: w_ind*papers_worker+papers_worker],
#                             papers_ids[w_ind*papers_worker: w_ind*papers_worker+papers_worker]):
#             cr_vals_id = range(p_id * criteria_num, p_id * criteria_num + criteria_num, 1)
#             isPaperIN = sum([GT[i] for i in cr_vals_id]) == 0
#             if isPaperIN:
#                 worker_acc = worker_acc_in
#             else:
#                 worker_acc = worker_acc_out
#
#             GT_cr = GT[p_id * criteria_num + cr]
#             cr_dif = criteria_difficulty[cr]
#             if cr_dif < 1.:
#                 if np.random.binomial(1, 0.55):
#                     vote = GT_cr
#                 else:
#                     vote = 1 - GT_cr
#             else:
#                 if np.random.binomial(1, worker_acc * cr_dif if worker_acc * cr_dif <= 1. else 1.):
#                     vote = GT_cr
#                 else:
#                     vote = 1 - GT_cr
#             responses.append(vote)
#     return responses