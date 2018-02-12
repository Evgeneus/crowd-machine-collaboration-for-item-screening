import numpy as np


def do_quiz_criteria_confm(quiz_papers_n, cheaters_prop, filters_dif):
    # decide if a worker a cheater
    if np.random.binomial(1, cheaters_prop):
        worker_type = 'rand_ch'
        worker_accuracy_out, worker_accuracy_in = 0.5, 0.5
    else:
        worker_type = 'worker'
        worker_accuracy_in = 0.5 + (np.random.beta(1, 1) * 0.5)
        worker_accuracy_out = worker_accuracy_in + 0.1 if worker_accuracy_in + 0.1 <= 1. else 1.

    for paper_id in range(quiz_papers_n):
        if np.random.binomial(1, 0.5):
            for mult in filters_dif:
                if worker_type == 'rand_ch':
                    if not np.random.binomial(1, worker_accuracy_in):
                        return [worker_type]
                elif not np.random.binomial(1, worker_accuracy_in*mult if worker_accuracy_in*mult <= 1. else 1.):
                    return [worker_type]
        else:
            for mult in [1., 1., 1.1, 0.9]:
                if worker_type == 'rand_ch':
                    if not np.random.binomial(1, worker_accuracy_out):
                        return [worker_type]
                elif not np.random.binomial(1, worker_accuracy_out*mult if worker_accuracy_out*mult <= 1. else 1.):
                    return [worker_type]
    return [worker_accuracy_out, worker_accuracy_in, worker_type]
