from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
import numpy as np
import csv
from realdata_experiment.crowd_main import *
from itertools import combinations
import random



def _load_data(path):
    texts, labels, pmids = [], [], []
    csv_reader = csv.reader(open(path, 'r'))
    next(csv_reader)  # skip headers
    for r in csv_reader:
        pmid, label, text = r
        texts.append(text)
        labels.append(int(label))
        pmids.append(pmid)
    return texts, labels, pmids


def machineRun(balancing):
    texts1, labels, pmids1 = _load_data('../data/proton-beam-merged.csv')
    classifiers = {}
    labels = []
    texts = []
    pmids = []

    getcrowdvotequestion = crowd_main(0)  # change the label with first question label!
    for item in getcrowdvotequestion.keys():
        pmids.append(item)
    for item in pmids:
        labels.append(getcrowdvotequestion[item])
    for item in pmids:
        index = pmids1.index(item)
        texts.append(texts1[index])

    if (balancing > 0):
        Outscope = [i for i, j in list(enumerate(labels)) if j == 0]  # get index
        Inscope = [i for i, j in list(enumerate(labels)) if j == 1]  # get index
        sample = len(Inscope) * balancing
        candid = random.sample(Outscope, sample)  # random sample from out
        texts = [j for i, j in list(enumerate(texts)) if i in Inscope] + [j for i, j in list(enumerate(texts)) if
                                                                          i in candid]
        labels = [j for i, j in list(enumerate(labels)) if i in Inscope] + [j for i, j in list(enumerate(labels)) if
                                                                            i in candid]
        pmids = [j for i, j in list(enumerate(pmids)) if i in Inscope] + [j for i, j in list(enumerate(pmids)) if
                                                                          i in candid]


    vectorizer = TfidfVectorizer(stop_words="english", min_df=3, max_features=50000, norm='l2')
    X = vectorizer.fit_transform(texts)
    X = X.toarray()
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.5)
    result = []

    # Machine 1 DummyClassifier
    print ('DummyClassifier_stratified')
    Random_classifier = DummyClassifier(strategy='stratified', random_state=42).fit(X_train, y_train)
    y_pred = Random_classifier.predict(X_test)
    classifiers['0'] = y_pred
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    accuracy_train = Random_classifier.score(X_train, y_train)
    accuracy_test = Random_classifier.score(X_test, y_test)
    f1score = metrics.f1_score(y_test, y_pred, average='macro')
    roc = metrics.roc_auc_score(y_test, y_pred)
    result.append(['DumClassifierStratified', accuracy_train, accuracy_test, f1score, roc, precision, recall])
    print ('accuracy_train:' + str(accuracy_train))
    print ('accuracy_test:' + str(accuracy_test))
    print ('f1score:' + str(f1score))
    print ('roc_auc_score:' + str(roc))
    print ('recall:' + str(recall))
    print ('precision:' + str(precision))

    # Machine 1 DummyClassifier
    print ('DummyClassifier_stratified')
    Random1_classifier = DummyClassifier(strategy='most_frequent', random_state=42).fit(X_train, y_train)
    y_pred = Random1_classifier.predict(X_test)
    classifiers['1'] = y_pred
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    accuracy_train = Random1_classifier.score(X_train, y_train)
    accuracy_test = Random1_classifier.score(X_test, y_test)
    f1score = metrics.f1_score(y_test, y_pred, average='macro')
    roc = metrics.roc_auc_score(y_test, y_pred)
    result.append(['DumClassifierMostfrequent', accuracy_train, accuracy_test, f1score, roc, precision, recall])
    print ('accuracy_train:' + str(accuracy_train))
    print ('accuracy_test:' + str(accuracy_test))
    print ('f1score:' + str(f1score))
    print ('roc_auc_score:' + str(roc))
    print ('recall:' + str(recall))
    print ('precision:' + str(precision))

    # Machine 1 NaiveBase
    print ('Machine 1 MultinomialNaiveBase')
    gs_NaiveBase_clf = MultinomialNB().fit(X_train, y_train)
    y_pred = gs_NaiveBase_clf.predict(X_test)
    classifiers['2'] = y_pred
    accuracy_train = gs_NaiveBase_clf.score(X_train, y_train)
    accuracy_test = gs_NaiveBase_clf.score(X_test, y_test)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1score = metrics.f1_score(y_test, y_pred, average='macro')
    roc = metrics.roc_auc_score(y_test, y_pred)
    result.append(['MultinomialNB', accuracy_train, accuracy_test, f1score, roc, precision, recall])
    print ('accuracy_train:' + str(accuracy_train))
    print ('accuracy_test:' + str(accuracy_test))
    print ('f1score:' + str(f1score))
    print ('roc_auc_score:' + str(roc))
    print ('recall:' + str(recall))
    print ('precision:' + str(precision))

    # Machine 1 BeNaiveBase
    print ('Machine 1 BernoulliNB')
    gs_NaiveBase_clf = BernoulliNB().fit(X_train, y_train)
    y_pred = gs_NaiveBase_clf.predict(X_test)
    classifiers['3'] = y_pred
    accuracy_train = gs_NaiveBase_clf.score(X_train, y_train)
    accuracy_test = gs_NaiveBase_clf.score(X_test, y_test)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1score = metrics.f1_score(y_test, y_pred, average='macro')
    roc = metrics.roc_auc_score(y_test, y_pred)
    result.append(['BernoulliNB', accuracy_train, accuracy_test, f1score, roc, precision, recall])
    print ('accuracy_train:' + str(accuracy_train))
    print ('accuracy_test:' + str(accuracy_test))
    print ('f1score:' + str(f1score))
    print ('roc_auc_score:' + str(roc))
    print ('recall:' + str(recall))
    print ('precision:' + str(precision))

    # Machine 2 SGD Norm2
    print ('Machine 2 SGD')
    params_d = {"alpha": 10.0 ** -np.arange(1, 7)}
    sgd = SGDClassifier(class_weight={1: 2}, random_state=42, penalty='l2')
    clfsgd = GridSearchCV(sgd, params_d, scoring='roc_auc', cv=3)
    clfsgd = clfsgd.fit(X_train, y_train)
    y_pred = clfsgd.predict(X_test)
    classifiers['4'] = y_pred
    accuracy_train = clfsgd.score(X_train, y_train)
    accuracy_test = clfsgd.score(X_test, y_test)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1score = metrics.f1_score(y_test, y_pred, average='macro')
    roc = metrics.roc_auc_score(y_test, y_pred)
    result.append(['SGDl2{1:2}', accuracy_train, accuracy_test, f1score, roc, precision, recall])
    print ('accuracy_train:' + str(accuracy_train))
    print ('accuracy_test:' + str(accuracy_test))
    print ('f1score:' + str(f1score))
    print ('roc_auc_score:' + str(roc))
    print ('recall:' + str(recall))
    print ('precision:' + str(precision))

    # Machine 2 SGD Norm1
    print ('Machine 3 SGD')
    sgd = SGDClassifier(class_weight={1: 1}, random_state=42, penalty='l1')
    clfsgd = GridSearchCV(sgd, params_d, scoring='roc_auc', cv=3)
    clfsgd = clfsgd.fit(X_train, y_train)
    y_pred = clfsgd.predict(X_test)
    classifiers['5'] = y_pred
    accuracy_train = clfsgd.score(X_train, y_train)
    accuracy_test = clfsgd.score(X_test, y_test)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1score = metrics.f1_score(y_test, y_pred, average='macro')
    roc = metrics.roc_auc_score(y_test, y_pred)
    result.append(['SGDl1{1:1}', accuracy_train, accuracy_test, f1score, roc, precision, recall])
    print ('accuracy_train:' + str(accuracy_train))
    print ('accuracy_test:' + str(accuracy_test))
    print ('f1score:' + str(f1score))
    print ('roc_auc_score:' + str(roc))
    print ('recall:' + str(recall))
    print ('precision:' + str(precision))

    # Machine 3 RandomForrest

    print ('Machine 4 RandomForrest')
    RF_clf = RandomForestClassifier(class_weight={1: 5}, random_state=42)
    parameters_RF = {
        'n_estimators': [300],  # 300 is enough
        'max_depth': [20]  # this is good fit
    }

    gs_RF_clf = GridSearchCV(RF_clf, parameters_RF, n_jobs=-1, scoring='roc_auc', cv=3)
    gs_RF_clf = gs_RF_clf.fit(X_train, y_train)
    print ('RF fitted!')
    y_pred = gs_RF_clf.predict(X_test)
    classifiers['6'] = y_pred
    accuracy_train = gs_RF_clf.score(X_train, y_train)
    accuracy_test = gs_RF_clf.score(X_test, y_test)
    f1score = metrics.f1_score(y_test, y_pred, average='macro')
    roc = metrics.roc_auc_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    result.append(['RF{1:5}', accuracy_train, accuracy_test, f1score, roc, precision, recall])
    print ('accuracy_train:' + str(accuracy_train))
    print ('accuracy_test:' + str(accuracy_test))
    print ('f1score:' + str(f1score))
    print ('roc_auc_score:' + str(roc))
    print ('recall:' + str(recall))
    print ('precision:' + str(precision))
    #
    # Machine 4 KNN
    print ('Machine 5 KNN')
    knn_clf = KNeighborsClassifier(weights='uniform')

    parameters_knn = {
        'n_neighbors': [2, 3, 4]
    }
    gs_knn_clf = GridSearchCV(knn_clf, parameters_knn, scoring='roc_auc', n_jobs=-1, cv=3)
    gs_knn_clf = gs_knn_clf.fit(X_train, y_train)
    y_pred = gs_knn_clf.predict(X_test)
    accuracy_train = gs_knn_clf.score(X_train, y_train)
    accuracy_test = gs_knn_clf.score(X_test, y_test)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1score = metrics.f1_score(y_test, y_pred)
    roc = metrics.roc_auc_score(y_test, y_pred)
    result.append(['KNN', accuracy_train, accuracy_test, f1score, roc, precision, recall])
    print ('accuracy_train:' + str(accuracy_train))
    print ('accuracy_test:' + str(accuracy_test))
    print ('f1score:' + str(f1score))
    print ('roc_auc_score:' + str(roc))
    print ('recall:' + str(recall))
    print ('precision:' + str(precision))
    # #
    # # Machine 4 GB
    print ('Machine 6 GB')
    GB_clf = GradientBoostingClassifier(random_state=42, max_features=0.1)

    parameters_GB = {
        'n_estimators': [200],
        'learning_rate': [0.1]

    }

    gb_clf = GridSearchCV(GB_clf, parameters_GB, scoring='roc_auc', n_jobs=-1, cv=3)
    gb_clf = gb_clf.fit(X_train, y_train)
    print ('GB fitted!')
    y_pred = gb_clf.predict(X_test)
    classifiers['7'] = y_pred
    accuracy_train = gb_clf.score(X_train, y_train)
    accuracy_test = gb_clf.score(X_test, y_test)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1score = metrics.f1_score(y_test, y_pred, average='macro')
    roc = metrics.roc_auc_score(y_test, y_pred)
    result.append(['GB', accuracy_train, accuracy_test, f1score, roc, precision, recall])
    print ('accuracy_train:' + str(accuracy_train))
    print ('accuracy_test:' + str(accuracy_test))
    print ('f1score:' + str(f1score))
    print ('roc_auc_score:' + str(roc))
    print ('recall:' + str(recall))
    print ('precision:' + str(precision))

    # Machine 4 GB


    print ('Machine 6 baggingWithSVC')
    n_estimators = 10
    SVC_clf = BaggingClassifier(base_estimator=SVC(kernel='linear', class_weight={1: 10}), n_estimators=n_estimators,
                                max_samples=1.0 / n_estimators, random_state=42, max_features=0.3)

    SVC_clf = SVC_clf.fit(X_train, y_train)
    print ('baggingWithSVC fitted!')

    y_pred = SVC_clf.predict(X_test)
    classifiers['8'] = y_pred
    accuracy_train = SVC_clf.score(X_train, y_train)
    accuracy_test = SVC_clf.score(X_test, y_test)
    f1score = metrics.f1_score(y_test, y_pred, average='macro')
    roc = metrics.roc_auc_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    result.append(['SVCBagging{1:10}', accuracy_train, accuracy_test, f1score, roc, precision, recall])
    print ('accuracy_train:' + str(accuracy_train))
    print ('accuracy_test:' + str(accuracy_test))
    print ('f1score:' + str(f1score))
    print ('roc_auc_score:' + str(roc))
    print ('recall:' + str(recall))
    print ('precision:' + str(precision))
    print ('*******************************')

    return result, classifiers, y_test


def correlation(classifiers, y_true):
    correlation_result = {}
    machine_combinations = []
    for i in range(2, 3):
        machine_combinations.append(list(combinations(classifiers.keys(), i)))  # combination of classifier
    for mac_combination in machine_combinations:
        for dual_combination in mac_combination:
            vote_list = []
            key = ""
            for machine in dual_combination:
                temp_resut = []

                for a, b in zip(classifiers[machine], y_true):
                    if a == b:
                        temp_resut.append(1)
                    else:
                        temp_resut.append(0)
                vote_list.append(temp_resut)

                key += str(machine)

            same_occurance = [sum(x) for x in zip(*vote_list)]

            both_false = len([x for x in same_occurance if x == 0])
            both_true = len([x for x in same_occurance if x == len(vote_list)])
            disagree = [i for i, x in enumerate(same_occurance) if x == 1]
            b = len([i for i in disagree if vote_list[0][i] == 1])
            c = len([i for i in disagree if vote_list[0][i] == 0])
            kappa = (2 * ((both_false * both_true) - (b * c))) / float((((both_true + b) * (b + both_false)) + ((both_true + c) * (both_false + c))))

            correlation_result[key] = [both_false, both_true, kappa]

    return correlation_result