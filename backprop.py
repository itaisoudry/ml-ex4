from __future__ import division
import data
import network
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy.random
import numpy as np


def b():
    training_data, test_data = data.load(train_size=10000, test_size=5000)
    net = network.Network([784, 40, 10])
    epochs = list(range(0, 30))
    test_accuracy, training_accuracy, training_loss = net.SGD(training_data, 30, 10, 0.1, test_data, False)

    plt.plot(epochs, test_accuracy)
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    plt.savefig('2bTestAcc.png')
    plt.clf()

    plt.plot(epochs, training_accuracy)
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.savefig('2bTrainAcc.png')
    plt.clf()

    plt.plot(epochs, training_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.savefig('2bTrainLoss.png')
    plt.clf()


def c():
    training_data, test_data = data.load(train_size=50000, test_size=10000)
    net = network.Network([784, 40, 10])
    net.SGD(training_data, 30, 10, 0.1,
            test_data, False)
    test_func = net.one_label_accuracy(test_data)
    print(test_func)


def d():
    training_data, test_data = data.load(train_size=10000, test_size=5000)
    net = network.Network([784, 30, 30, 30, 30, 10])
    test_accuracy, training_accuracy, training_loss, norm_bis = net.SGD(training_data, 30, 10000, 0.1, test_data, True)
    new_epochs = np.arange(30).reshape((30, 1))
    colors = ["red", "green", "blue", "orange", "purple"]

    for i in range(net.num_layers - 1):
        plt.scatter(new_epochs, norm_bis[:, i].reshape((30, 1)), color=colors[i])

    plt.xlabel("Epoch")
    plt.ylabel("Norm")
    r_patch = mpatches.Patch(color='red', label='b1')
    b_patch = mpatches.Patch(color='green', label='b2')
    g_patch = mpatches.Patch(color='blue', label='b3')
    y_patch = mpatches.Patch(color='orange', label='b4')
    m_patch = mpatches.Patch(color='purple', label='b5')
    plt.legend(handles=[r_patch, b_patch, g_patch, y_patch, m_patch])
    plt.legend()
    plt.savefig("2d.png")
    plt.clf()


# b()
# c()
d()
