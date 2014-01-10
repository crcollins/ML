import csv
import itertools
import multiprocessing
import ast

import numpy

data = []
with open("data_clean.csv", "r") as csvfile:
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

features = []
for i, (name, feat, occ, virt, orb, dip, eng, gap, time) in enumerate(data):
    features.append(feat)
    HOMO[i] = occ
    LUMO[i] = virt
    DIPOLE[i] = dip
    ENERGY[i] = eng
    GAP[i] = gap

FEATURES = []
for group in zip(*tuple(features)):
    FEATURES.append(numpy.matrix(group))


def get_weight(X, y, limit=400):
    Xin = X[:limit,  :]
    Xout = X[limit:, :]
    yin = y[:limit,  :]
    yout = y[limit:, :]
    W = numpy.linalg.pinv(Xin.T*Xin)*Xin.T*yin
    error = numpy.abs(yin-Xin*W).mean()
    cross_error = numpy.abs(yout-Xout*W).mean()
    return W, error, cross_error

def get_learning_curves(X, y, step=25):
    M, N = X.shape
    for lim in xrange(50, M, step):
        W, e1, e2 = get_weight(X, y, limit=lim)
        print lim, e1, e2

def get_high_errors(errors, limit=1.5):
    aerrors = numpy.abs(errors)
    mean = aerrors.mean()
    std = aerrors.std()
    results = []
    for x in aerrors.argsort(0)[::-1]:
        if aerrors[x[0,0]] > (mean + limit * std):
            results.append((data[x[0,0]][0], aerrors[x[0,0]][0,0]))
    return results


class Linear(object):
    def __init__(self):
        self.weights = None
    def fit(self, X, y):
        X = numpy.matrix(X)
        y = numpy.matrix(y).T
        self.weights = numpy.linalg.pinv(X.T*X)*X.T*y
    def predict(self, X):
        X = numpy.matrix(X)
        return X*self.weights


from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from sklearn import dummy
from sklearn import neighbors

from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection

def test_clf(X, y, clf, test_size=0.2, num=20):
    ylist = y.T.tolist()[0]
    train = numpy.zeros(num)
    cross = numpy.zeros(num)
    for i in xrange(num):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, ylist, test_size=test_size)
        clf.fit(X_train, y_train)
        train[i] = mean_absolute_error(clf.predict(X_train), y_train)
        cross[i] = mean_absolute_error(clf.predict(X_test), y_test)
    return (train.mean(), train.std()), (cross.mean(), cross.std())


def test_clf_kfold(X, y, clf, folds=10):
    train = numpy.zeros(folds)
    cross = numpy.zeros(folds)
    for i, (train_idx, test_idx) in enumerate(cross_validation.KFold(y.shape[0], n_folds=folds)):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx].T.tolist()[0]
        y_test = y[test_idx].T.tolist()[0]
        clf.fit(X_train, y_train)
        train[i] = mean_absolute_error(clf.predict(X_train), y_train)
        cross[i] = mean_absolute_error(clf.predict(X_test), y_test)
    return (train.mean(), train.std()), (cross.mean(), cross.std())


def test_sklearn(X, y):
    funcs = {
        "dummy": dummy.DummyRegressor(),
        "linear": Linear(),
        "neighbors": neighbors.KNeighborsRegressor(),
        "linear ridge .05": linear_model.Ridge(alpha = .05),
        "linear ridge .5": linear_model.Ridge(alpha = .5),
        "linear ridge 5": linear_model.Ridge(alpha = 5),
        # "LARS .01": linear_model.LassoLars(alpha=.01),
        # "LARS .1": linear_model.LassoLars(alpha=.1),
        # "LARS 1": linear_model.LassoLars(alpha=1),
        # "svm": svm.SVR(),
        # "svm rbf": svm.SVR(kernel='rbf'),
        "svm rbf 2": svm.SVR(C=0.1, kernel="rbf", gamma=0.1),
        "svm rbf 3": svm.SVR(C=20, kernel="rbf", gamma=0.1),
        "svm rbf 4": svm.SVR(C=10, kernel="rbf", gamma=0.05),
        "tree": tree.DecisionTreeRegressor(),
        # "tree 1": tree.DecisionTreeRegressor(max_depth=1),
        # "tree 10": tree.DecisionTreeRegressor(max_depth=10),
        # "tree 100": tree.DecisionTreeRegressor(max_depth=100),
    }
    results = {}
    for name, clf in funcs.items():
        print name
        train = []
        test = []
        for val in xrange(1, 9):
            a,b = test_clf(X, y, clf, test_size=val*0.1)
        for val in xrange(1, 2):
            # a,b = test_clf(X, y, clf, test_size=val*0.1, num=100)
            a,b = test_clf_kfold(X, y, clf, folds=10)
            train.append(a)
            test.append(b)
        results[name] = (train, test)
    return results

def scan(X, y, function, params):
    size = [len(x) for x in params.values()]
    train_results = numpy.zeros(size)
    test_results = numpy.zeros(size)
    keys = params.keys()
    values = params.values()
    for group in itertools.product(*values):
        idx = tuple([a.index(b) for a,b in zip(values, group)])
        a = dict(zip(keys, group))
        clf = function(**a)
        print a, idx
        train, test = test_clf(X, y, clf)
        train_results[idx] = train[0]
        test_results[idx] = test[0]
    return train_results, test_results


def sort_xset(xset):
    temp = {}
    for yset in xset:
        for name in yset:
            zipped = zip(*yset[name])
            (_, test) = zipped[0]
            if temp.get(name):
                temp[name].append(test[0])
            else:
                temp[name] = [test[0]]
    for key, val in temp.items():
        temp[key] = sum(val)/len(val)
    return [y for y in sorted(temp, key=lambda x: temp[x])]


def display_sorted_results(results):
    for xset in results:
        order = sort_xset(xset)
        for name in order:
            print '"' + name + '"'
            lines = []
            for yset in xset:
                for i, (train, test) in enumerate(zip(*yset[name])):
                    try:
                        lines[i].extend([train, test])
                    except IndexError:
                        lines.append([train, test])
            for line in lines:
                tests = line[1::2]
                means = [x[0] for x in tests]
                stds = [x[1] for x in tests]
                means.append(sum(means)/len(means))
                stds.append(sum(stds)/len(stds))
                spacers = ['', '', '', '']
                print ','.join(str(x) for x in sum(zip(spacers, means, stds), ()))
        print '\n'



def gauss_kernel_gen(sigma):
    def func(X, Y):
        return numpy.exp(sigma*-cdist(X,Y)**2)
    return func


def laplace_kernel_gen(sigma):
    def func(X, Y):
        return numpy.exp(sigma*-cdist(X,Y))
    return func

def power_kernel_gen(sigma, p):
    def func(X, Y):
        return numpy.exp(sigma*-cdist(X,Y)**p)
    return func

def multi_func(xset):
    temp = []
    for yset in (HOMO, LUMO, GAP):
        temp.append(test_sklearn(xset, yset))
    return temp


def main():
    pool = multiprocessing.Pool(processes=4)
    results = pool.map(multi_func, FEATURES)
    display_sorted_results(results)
    return results


def run_nn(X, y, NN=None, test_size=0.1):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y.T.tolist()[0], test_size=.1)
    if NN is None:
        NN = (153, 600, 300, 1)
    net = buildNetwork(*NN, bias=True)
    ds = SupervisedDataSet(X.shape[1], 1)
    for i, row in enumerate(X_train):
        ds.addSample(tuple(row), y_train[i])
    trainer = BackpropTrainer(net, ds)
    for i in xrange(250):
        trainer.train()
        r = []
        for row in X_test.tolist(): r.append(net.activate(row)[0])
        results = numpy.matrix(r)
        r2 = []
        for row in X_train.tolist(): r2.append(net.activate(row)[0])
        results2 = numpy.matrix(r2)
        print i, numpy.abs(results2-y_train).mean(), numpy.abs(results-y_test).mean()
    return net, ds, trainer



class NeuralNet(object):
    def __init__(self, hidden_layers=None):
        self.hidden_layers = list(hidden_layers)

    def build_network(self, layers=None):
        layerobjects = []
        for item in layers:
            try:
                t, n = item
                if t == "sig":
                    if n == 0:
                        continue
                    layerobjects.append(SigmoidLayer(n))
            except TypeError:
                layerobjects.append(LinearLayer(item))

        n = FeedForwardNetwork()
        n.addInputModule(layerobjects[0])

        for i, layer in enumerate(layerobjects[1:-1]):
            n.addModule(layer)
            connection = FullConnection(layerobjects[i], layerobjects[i+1])
            n.addConnection(connection)

        n.addOutputModule(layerobjects[-1])
        connection = FullConnection(layerobjects[-2], layerobjects[-1])
        n.addConnection(connection)

        n.sortModules()
        return n

    def fit(self, X, y):
        n = X.shape[1]
        self.nn = self.build_network([n]+self.hidden_layers+[1])
        ds = SupervisedDataSet(n, 1)
        for i, row in enumerate(X):
            ds.addSample(row.tolist(), y[i])
        trainer = BackpropTrainer(self.nn, ds)
        for i in xrange(100):
            trainer.train()

    def predict(self, X):
        r = []
        for row in X.tolist():
            r.append(self.nn.activate(row)[0])
        return numpy.array(r)


# clf = NeuralNet([("sig", 250), ("sig", 250)])
# print test_clf(FEATURES[0], GAP, clf, num=1)

# layers = []
# from itertools import product
# vals = [0, 10, 50, 100, 200, 400]
# possible_layers = set(tuple(x for x in vals if x) for vals in product(vals, vals, vals, vals))
# print len(possible_layers)

results = main()

def func(layers):
    print layers
    new = zip(["sig"]*len(layers), layers)
    clf = NeuralNet(new)
    res = test_clf(FEATURES[0], GAP, clf, num=1)
    print res
    return res

# def func(num): return test_clf(FEATURES[0], GAP, NeuralNet([("sig", x)]), num=1)[1][0]


# pool = multiprocessing.Pool(processes=4)
# results = pool.map(func, possible_layers)

# for i, layers in enumerate(possible_layers):
#     new = zip(["sig"]*len(layers), layers)
#     clf = NeuralNet(new)
#     print i, layers,
#     print test_clf(FEATURES[0], GAP, clf, num=1)




# clf.fit(FEATURES[0], GAP.T.tolist()[0])
# print clf.predict(FEATURES[0])
# results = main()


import matplotlib.pyplot as plt
from sklearn import decomposition

def PCA_stuff(X, y, title="Principal Component Analysis"):
    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    print pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_)
    temper = pca.transform(X)
    Xs = temper[:,0]
    Ys = temper[:,1]
    COLOR = (y-y.min())/y.max()
    cm = plt.get_cmap("HOT")
    plt.scatter(Xs, Ys, c=COLOR,s=80, marker='o', edgecolors='none')
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()
    plt.clf()

def PCA_stuff_3d(X, y, title="Principal Component Analysis"):
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    print pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_)
    temper = pca.transform(X)
    Xs = temper[:,0]
    Ys = temper[:,1]
    Zs = temper[:,2]
    COLOR = (y-y.min())/y.max()
    cm = plt.get_cmap("HOT")
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xs, Ys, Zs, c=COLOR,s=80, marker='o', edgecolors='none')
    plt.title(title)
    plt.show()
    plt.clf()

