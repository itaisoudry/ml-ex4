from __future__ import division

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy.random
import sklearn.preprocessing
from numpy import *
from sklearn.datasets import fetch_mldata

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


def get_logscale():
    scale = []
    i = -3
    while i <= 3:
        scale.append(double(pow(10, i)))
        i += 0.5
    return scale


def sgd(samples, labels, C, n0, T):
    n, d = samples.shape
    w = numpy.zeros(d, numpy.float64)
    norm_samples = numpy.divide(samples, numpy.linalg.norm(samples))

    for t in range(1, T):
        # sample i uniformly
        i = numpy.random.randint(0, len(samples), 1)[0]

        label = labels[i]
        sample = norm_samples[i]
        nt = n0 / t
        res = label * numpy.dot(w, sample)

        if res < 1:
            #  (1 − ηt)wt + ηtCyixi
            w = numpy.multiply((1 - nt), w) + numpy.multiply(nt * C * label, sample)
        else:
            w = numpy.multiply((1 - nt), w)

    return w


def cross_validate(classifier, val_data, val_labels):
    errors = 0
    val_data_norm = numpy.divide(val_data, numpy.linalg.norm(val_data))
    for i in range(len(val_data)):
        data = val_data_norm[i]
        label = val_labels[i]
        sign = numpy.sign(numpy.dot(classifier, data))
        # print(sign)
        if sign != label:
            errors += 1
    return (len(val_data) - errors) / len(val_data)


def a():
    avg_accuracies = []
    ns = []
    for i in range(-5, 6):
        n = pow(10, i)
        ns.append(n)

        accuracy = 0
        # cross validation
        for i in range(10):
            # train svm
            classifier = sgd(train_data, train_labels, 1, n, 1000)
            accuracy += cross_validate(classifier, validation_data, validation_labels)

        avg_accuracies.append(accuracy / 10)

    # plot
    plt.scatter(ns, avg_accuracies)
    plt.xlabel('ni_0')
    plt.ylabel('Avg. Accuracy')
    plt.savefig('1a.png')
    plt.clf()

    # return best n0
    return pow(10, avg_accuracies.index(max(avg_accuracies)) - 5)


def b(n0):
    avg_accuracies = []
    cs = []
    for i in range(-5, 6):
        c = pow(10, i)
        cs.append(c)
        accuracy = 0

        # cross validation
        for i in range(10):
            classifier = sgd(train_data, train_labels, c, n0, 1000)
            accuracy += cross_validate(classifier, validation_data, validation_labels)

        avg_accuracies.append(accuracy / 10)

    # plot
    axes = plt.gca()
    axes.set_xlim(min(cs), max(cs))
    axes.set_ylim(min(avg_accuracies), max(avg_accuracies))
    plt.scatter(cs, avg_accuracies)
    plt.xlabel('C')
    plt.ylabel('Avg. Accuracy')
    plt.savefig('1b.png')
    plt.clf()
    # return best C
    return pow(10, avg_accuracies.index(max(avg_accuracies)) - 5)


def c(n0, c):
    classifier = sgd(train_data, train_labels, c, n0, 20000)
    plt.imshow(reshape(classifier, (28, 28)), interpolation='nearest')
    plt.savefig('1c.png')
    plt.clf()
    accuracy = cross_validate(classifier, test_data, test_labels)
    print('Best classifier accuracy on the test set:', accuracy)


n0 = a()
print('best n0:', n0)
C = b(n0)
print('best c:', C)
c(n0, C)
