import csv
import itertools
import multiprocessing
import ast

import numpy
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import scipy.optimize

from sklearn import decomposition
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


###############################################################################
# Load Data
###############################################################################


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



###############################################################################
# CLFs
###############################################################################


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

class KernelRegression(object):
    def __init__(self, kernel=None, gamma=0.05, C=10):
        self.weights = None
        self.kernel = gauss_kernel_gen(gamma)
        self.X = None
        self.C = C
        self.gamma = gamma
    def fit(self, X, y):
        X = numpy.matrix(X)
        y = numpy.matrix(y).T
        self.X = X
        print self.kernel(X, X).shape, (self.C*numpy.eye(X.shape[0])).shape
        self.weights = numpy.linalg.pinv(self.kernel(X, X)+self.C*numpy.eye(X.shape[0]))*y
    def predict(self, X):
        X = numpy.matrix(X)
        return self.kernel(X, self.X)*self.weights


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


class SVMLaplace(svm.SVR):
    def __init__(self, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, tol=1e-3,
                 C=1.0, epsilon=0.1, shrinking=True, probability=False,
                 cache_size=200, verbose=False, max_iter=-1,
                 random_state=None):
        super(SVMLaplace, self).__init__(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol,
                 C=C, epsilon=epsilon, shrinking=shrinking, probability=probability,
                 cache_size=cache_size, verbose=verbose, max_iter=max_iter,
                 random_state=random_state)
        self.kernel = laplace_kernel_gen(gamma)


###############################################################################
# Kernels
###############################################################################

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


##########################################
# Information
##########################################


def get_high_errors(errors, limit=1.5):
    aerrors = numpy.abs(errors)
    mean = aerrors.mean()
    std = aerrors.std()
    results = []
    for x in aerrors.argsort(0)[::-1]:
        if aerrors[x[0,0]] > (mean + limit * std):
            results.append((data[x[0,0]][0], aerrors[x[0,0]][0,0]))
    return results

def scan(X, y, function, params):
    size = [len(x) for x in params.values()]
    train_results = numpy.zeros(size)
    test_results = numpy.zeros(size)
    keys = params.keys()
    values = params.values()
    for group in itertools.product(*values):
        idx = tuple([a.index(b) for a,b in zip(values, group) if len(a) > 1])
        a = dict(zip(keys, group))
        clf = function(**a)
        train, test = test_clf_kfold(X, y, clf)
        # print a, idx, train, test
        train_results[idx] = train[0]
        test_results[idx] = test[0]
    return train_results, test_results


class OptimizedCLF(object):
    def __init__(self, X, y, func, params):
        self.params = params
        self.func = func
        self.X = X
        self.y = y
        self.optimized_clf = None
        self.optimized_params = None

    def __call__(self, *args):
        a = dict(zip(self.params.keys(), *args))
        clf = self.func(**a)
        train, test = test_clf_kfold(self.X, self.y, clf, folds=5)
        return test[0]

    def get_optimized_clf(self):
        if not len(self.params.keys()):
            self.optimized_clf = self.func()
        if self.optimized_clf is not None:
            return self.optimized_clf
        listparams = dict((k,v) for k,v in self.params.items() if type(v) in [list, tuple])
        itemparams = dict((k,v) for k,v in self.params.items() if type(v) not in [list, tuple])
        listvalues = []
        itemvalues = []
        if listparams:
            _, test = scan(self.X, self.y, self.func, listparams)
            listvalues = []
            temp = numpy.unravel_index(test.argmin(), test.shape)
            for i, pick in enumerate(listparams.values()):
                listvalues.append(pick[temp[i]])
            listvalues = listvalues[::-1]
        if itemparams:
            bounds = ((1e-8, None), ) * len(self.params.keys())
            results = scipy.optimize.fmin_l_bfgs_b(
                self, self.params.values(),
                bounds=bounds,
                approx_grad=True, epsilon=1e-3)
            itemvalues = results[0].tolist()
        keys = listparams.keys() + itemparams.keys()
        values = listvalues + itemvalues
        self.optimized_params = dict(zip(keys, values))
        self.optimized_clf = self.func(**self.optimized_params)
        return self.optimized_clf


##########################################
# Test
##########################################


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
        "dummy": (dummy.DummyRegressor, dict()),
        "linear": (Linear, dict()),
        "linear ridge": (linear_model.Ridge, {"alpha": 0.5}),
        "neighbors": (neighbors.KNeighborsRegressor, {"n_neighbors": [2, 3, 5, 10]}),
        "svm gauss": (svm.SVR, {"C": 10, "gamma": 0.05}),
        "svm laplace": (SVMLaplace, {"C": 10, "gamma": 0.05}),
        "tree": (tree.DecisionTreeRegressor, {"max_depth": [1, 5, 10, 50, None]}),
    }
    results = {}
    for name, (func, params) in funcs.items():
        print name
        train = []
        test = []
        for val in xrange(1, 2):
            clf = OptimizedCLF(X, y, func, params).get_optimized_clf()
            a,b = test_clf_kfold(X, y, clf, folds=10)
            train.append(a)
            test.append(b)
        if name == "svm laplace":
            clf.kernel = "laplace"
        results[name] = (train, test, clf)
    return results


###############################################################################
# Display
###############################################################################


def sort_xset(xset):
    temp = {}
    for yset in xset:
        for name in yset:
            zipped = zip(*yset[name][:-1])
            (_, test) = zipped[0]
            if temp.get(name):
                temp[name].append(test[0])
            else:
                temp[name] = [test[0]]
    for key, val in temp.items():
        temp[key] = sum(val)/len(val)
    return [y for y in sorted(temp, key=lambda x: temp[x])]


def display_sorted_results(results):
    clfs = []
    for xset in results:
        order = sort_xset(xset)
        for name in order:
            print '"' + name + '"',
            lines = []
            for yset in xset:
                clfs.append(yset[name][-1])
                for i, (train, test) in enumerate(zip(*yset[name][:-1])):
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
    return clfs


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


def plot_num_samples(X, y, clf, steps=25):
    trainvals = []
    testvals = []
    xvals = [i*1.0/steps for i in xrange(1, steps)]
    for val in xvals:
        train, test = test_clf(X, y, clf, test_size=val, num=20)
        trainvals.append(train[0])
        testvals.append(test[0])
    plt.plot(xvals, trainvals, '--', label="Training")
    plt.plot(xvals, testvals, label="Test")
    plt.legend(loc="best")
    plt.xlabel("Percent in Training Set")
    plt.ylabel("MAE (eV)")
    plt.show()


def plot_scan(X, y, function, params):
    listform = params.items()
    train, test = scan(X, y, function, params)
    xvals = listform[0][1]
    plt.plot(numpy.log(xvals), train, '--', label="Training")
    plt.plot(numpy.log(xvals), test, label="Test")
    plt.legend(loc="best")
    plt.xlabel("log(%s)" % listform[0][0])
    plt.ylabel("MAE (eV)")
    plt.title("Optimization of %s" % listform[0][0])
    plt.show()


def plot_scan_2d(X, y, function, params):
    listform = params.items()
    train, test = scan(X, y, function, params)
    plt.matshow(test)
    plt.xlabel(listform[0][0])
    plt.ylabel(listform[1][0])
    plt.show()

def plot_homo_lumo(homo, lumo, gap, clf):
    HL = numpy.concatenate((lumo-homo, numpy.ones(homo.shape)),1)
    clf.fit(HL, gap.T.tolist()[0])
    pred = clf.predict(HL)
    lim = max(gap.max(), clf.predict(HL).max())
    std = (pred-gap).std()
    offset = ((std**2)/2)**0.5
    plt.plot(pred,gap,'b.')
    plt.plot([0,lim],[0,lim], 'r')
    plt.plot([0,lim],[offset,lim+offset], 'g--')
    plt.plot([0,lim],[-offset,lim-offset], 'g--')
    plt.xlabel("LUMO-HOMO (eV)")
    plt.ylabel("GAP (eV)")
    plt.title("Simple Prediction of Gap Value verses Gap Value")
    plt.show()

###############################################################################
# Neural Net
###############################################################################

# clf = NeuralNet([("sig", 250), ("sig", 250)])
# print test_clf(FEATURES[0], GAP, clf, num=1)

# layers = []
# from itertools import product
# vals = [0, 10, 50, 100, 200, 400]
# possible_layers = set(tuple(x for x in vals if x) for vals in product(vals, vals, vals, vals))
# print len(possible_layers)

# def func(layers):
#     print layers
#     new = zip(["sig"]*len(layers), layers)
#     clf = NeuralNet(new)
#     res = test_clf(FEATURES[0], GAP, clf, num=1)
#     print res
#     return res

# def func(num): return test_clf(FEATURES[0], GAP, NeuralNet([("sig", x)]), num=1)[1][0]


# pool = multiprocessing.Pool(processes=4)
# results = pool.map(func, possible_layers)

# for i, layers in enumerate(possible_layers):
#     new = zip(["sig"]*len(layers), layers)
#     clf = NeuralNet(new)
#     print i, layers,
#     print test_clf(FEATURES[0], GAP, clf, num=1)



###############################################################################
# Main
###############################################################################


def multi_func(xset):
    temp = []
    for yset in (HOMO, LUMO, GAP):
        temp.append(test_sklearn(xset, yset))
    return temp


def multi_func2(params):
    yset, xset = params
    return test_sklearn(xset, yset)


def main():
    pool = multiprocessing.Pool(processes=4)
    results = pool.map(multi_func, FEATURES)
    results2 = pool.map(multi_func2, FEATURES1)
    clfs = display_sorted_results(results)
    clfs2 = display_sorted_results(results2)
    return results, results2, clfs, clfs2

FEATURES1 = []
for i, feat in enumerate(FEATURES[1:]):
    homoclf = OptimizedCLF(feat, HOMO, svm.SVR, {"C": 10, "gamma": 0.05}).get_optimized_clf()
    homoclf.fit(feat, HOMO.T.tolist()[0])
    HOMOp = numpy.matrix(homoclf.predict(feat)).T

    lumoclf = OptimizedCLF(feat, LUMO, svm.SVR, {"C": 10, "gamma": 0.05}).get_optimized_clf()
    lumoclf.fit(feat, LUMO.T.tolist()[0])
    LUMOp = numpy.matrix(lumoclf.predict(feat)).T

    gapclf = OptimizedCLF(feat, GAP, svm.SVR, {"C": 10, "gamma": 0.05}).get_optimized_clf()
    gapclf.fit(feat, GAP.T.tolist()[0])
    GAPp = numpy.matrix(gapclf.predict(feat)).T

    FEATURES1.append((GAP, numpy.concatenate([feat, HOMOp, LUMOp], 1)))
    FEATURES1.append((HOMO, numpy.concatenate([feat, LUMOp, GAPp], 1)))
    FEATURES1.append((LUMO, numpy.concatenate([feat, GAPp, HOMOp], 1)))


if __name__ == "__main__":
    results = main()