import csv
import ast
import random
from itertools import product

import numpy
from scipy.optimize import curve_fit

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, TanhLayer
from pybrain.structure import FullConnection


class NeuralNet(object):
    def __init__(self, layers):
        self.layers = layers
        self.ds = None
        self.train_error = []
        self.test_error = []
        self.norm_error = []

    def improve(self, n=10):
        trainer = BackpropTrainer(self.nn, self.ds)
        for i in xrange(n):
            self.train_error.append(trainer.train())

    def fit(self, X, y):
        self.nn = buildNetwork(*self.layers, bias=True, hiddenclass=SigmoidLayer)

        self.ds = SupervisedDataSet(self.layers[0], self.layers[-1])
        for i, row in enumerate(X):
            self.ds.addSample(row.tolist(), y[i])
        self.improve()

    def predict(self, X):
        r = []
        for row in X.tolist():
            r.append(self.nn.activate(row))
        return numpy.array(r)


# clf = NeuralNet([("sig", 250), ("sig", 250)])
# print test_clf(FEATURES[0], GAP, clf, num=1)



# def func(layers):
#     print layers
#     new = zip(["sig"]*len(layers), layers)
#     clf = NeuralNet(new)
#     res = test_clf(FEATURES[0], GAP, clf, num=1)
#     print res
#     return res

# pool = multiprocessing.Pool(processes=4)
# results = pool.map(func, possible_layers)

# for i, layers in enumerate(possible_layers):
#     new = zip(["sig"]*len(layers), layers)
#     clf = NeuralNet(new)
#     print i, layers,
#     print test_clf(FEATURES[0], GAP, clf, num=1)






def power_reg(x, a, b):
    return a * x ** b

def fit_it(errors):
    x = numpy.arange(1,len(errors)+1)
    y = numpy.array(errors)
    (a, b), var_matrix = curve_fit(power_reg, x, y, p0=[1, -.5])
    return a, b

if __name__ == "__main__":
    data = []
    with open("cleaned_data.csv", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            temp = [row[3]]
            for item in row[4:]:
                try:
                    x = ast.literal_eval(item)
                    if x == []:
                        break
                    temp.append(x)
                except:
                    pass

            if len(temp) == 9:
                data.append(temp)

    M = len(data)

    HOMO = numpy.zeros((M, 1))
    LUMO = numpy.zeros((M, 1))
    DIPOLE = numpy.zeros((M, 1))
    ENERGY = numpy.zeros((M, 1))
    GAP = numpy.zeros((M, 1))
    TIME = numpy.zeros((M, 1))

    features = []
    for i, (name, feat, occ, virt, orb, dip, eng, gap, time) in enumerate(data):
        features.append(feat)
        HOMO[i] = occ
        LUMO[i] = virt
        DIPOLE[i] = dip
        ENERGY[i] = eng
        GAP[i] = gap
        TIME[i] = time
    YSETS = (HOMO, LUMO, GAP, ENERGY)

    FEATURES = []
    for group in zip(*tuple(features)):
        FEATURES.append(numpy.matrix(group))


    X = numpy.array(FEATURES[1])
    y = numpy.array(numpy.concatenate([HOMO, LUMO, GAP], 1))

    temp = range(len(X))
    random.shuffle(temp)
    X = X[temp,:]
    y = y[temp,:]

    split = int(.8 * X.shape[0])
    XTrain = X[:split, :]
    yTrain = y[:split, :]
    XTest = X[split:, :]
    yTest = y[split:, :]

    n = X.shape[1]
    if len(y.shape) > 1:
        m = y.shape[1]
    else:
        m = 1

    first = [10, 50, 100, 200, 400]
    second = [10, 50, 100]
    possible_layers = set((n, ) + tuple(x for x in vals if x) + (m, ) for vals in product(first, second, second))
    print len(possible_layers)
    possible_layers = list(possible_layers)[-10::-1]


    # def func(layers):
    for layers in ([n, 25, 15, m], ):
        print layers
        clf = NeuralNet(layers)
        clf.fit(XTrain, yTrain)
        clf.test_error.append(numpy.abs(clf.predict(XTest) - yTest).mean(0))
        clf.norm_error.append(numpy.linalg.norm(clf.test_error[-1]))
        print -1, clf.test_error[-1], clf.norm_error[-1]
        for i in xrange(100):
            clf.improve()
            clf.test_error.append(numpy.abs(clf.predict(XTest) - yTest).mean(0))
            clf.norm_error.append(numpy.linalg.norm(clf.test_error[-1]))
            temp = fit_it(clf.norm_error)
            print i, clf.test_error[-1], clf.norm_error[-1], temp[1]
        print
