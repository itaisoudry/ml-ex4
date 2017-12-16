from __future__ import division

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy.random
import sklearn.preprocessing
from numpy import *
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import cross_val_score

mnist = fetch_mldata('MNIST original', data_home='./')
data = mnist['data']
labels = mnist['target']

neg, pos = 0, 8
train_idx = numpy.random.RandomState(0).permutation(where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
test_idx = numpy.random.RandomState(0).permutation(where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

train_data_unscaled = data[train_idx[:6000], :].astype(float)
train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

validation_data_unscaled = data[train_idx[6000:], :].astype(float)
validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

test_data_unscaled = data[60000 + test_idx, :].astype(float)
test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

# Preprocessing
train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)


def sgd(samples, labels, C, n0, T):
    w = numpy.zeros(784)

    for t in range(1, T):
        # sample i uniformly
        i = numpy.random.randint(0, len(samples), 1)[0]
        label = labels[i]
        sample = samples[i]
        nt = n0 / t
        sign = label * numpy.dot(w, sample)
        errors = 0

        if sign < 1:
            #  (1 − ηt)wt + ηtCyixi
            w = numpy.multiply((1 - nt), w) + numpy.multiply(nt * C * label, sample)
        else:
            w = numpy.multiply((1 - nt), w)

        if sign != label:
            errors += 1
    return w, errors


def cross_validate(classifier, val_data, val_labels):
    errors = 0
    for i in range(len(val_data)):
        data = val_data[i]
        label = val_labels[i]
        if numpy.sign(numpy.dot(classifier,data)) != label:
            errors += 1
    return (len(val_data) - errors) / len(val_data)


def a():
    avg_accuracies = []
    ns = []
    for n in range(-5, 5):
        n0 = pow(10, n)
        ns.append(n0)
        # train svm
        classifier, errors = sgd(train_data, train_labels, 1, n0, 1000)

        accuracy = 0
        # cross validation
        for i in range(10):
            accuracy += cross_validate(classifier, validation_data, validation_labels)

        avg_accuracies.append(accuracy / 10)

    # plot
    plt.scatter(ns, avg_accuracies)
    plt.xlabel('ni_0')
    plt.ylabel('Avg. Accuracy')
    # plt.savefig('1a.png')
    plt.show()

    # plt.clf()


a()
