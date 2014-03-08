import multiprocessing
import cPickle

import numpy
from sklearn import decomposition
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from sklearn import dummy
from sklearn import neighbors

from clfs import Linear, SVMLaplace
from display import display_sorted_results, plot_scan_2d, plot_scan, PCA_stuff
from display import plot_num_samples
from utils import OptimizedCLF, test_clf_kfold


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


def get_extended_features(homo, lumo, gap, features):
    try:
        with open("features1.pkl", "rb") as f:
            extended_features = cPickle.load(f)
        if len(extended_features) == 4 * len(features):
            return extended_features
    except IOError:
        pass

    extended_features = []
    for i, feat in enumerate(features):
        print i
        print "homo"
        homoclf = OptimizedCLF(feat, homo, svm.SVR, {"C": 10, "gamma": 0.05}).get_optimized_clf()
        homoclf.fit(feat, homo.T.tolist()[0])
        HOMOp = numpy.matrix(homoclf.predict(feat)).T
        print "lumo"
        lumoclf = OptimizedCLF(feat, lumo, svm.SVR, {"C": 10, "gamma": 0.05}).get_optimized_clf()
        lumoclf.fit(feat, lumo.T.tolist()[0])
        LUMOp = numpy.matrix(lumoclf.predict(feat)).T
        print "gap"
        gapclf = OptimizedCLF(feat, gap, svm.SVR, {"C": 10, "gamma": 0.05}).get_optimized_clf()
        gapclf.fit(feat, gap.T.tolist()[0])
        GAPp = numpy.matrix(gapclf.predict(feat)).T

        extended_features.append((gap, numpy.concatenate([feat, HOMOp, LUMOp], 1)))
        extended_features.append((homo, numpy.concatenate([feat, LUMOp, GAPp], 1)))
        extended_features.append((lumo, numpy.concatenate([feat, GAPp, HOMOp], 1)))
        extended_features.append((gap, numpy.concatenate([HOMOp, LUMOp, numpy.ones(GAPp.shape)], 1)))
    with open("features1.pkl", "wb") as f:
        cPickle.dump(extended_features, f, protocol=-1)
    return extended_features


def multi_func(pair):
    temp = []
    feat, ysets = pair
    for yset in ysets:
        temp.append(test_sklearn(feat, yset))
    return temp


def extended_multi_func(params):
    yset, xset = params
    return [test_sklearn(xset, yset)]


def main(ysets, features, extended_features):
    temp = []
    for feat in features:
        temp.append((feat, ysets))

    pool = multiprocessing.Pool(processes=4)
    results = pool.map(multi_func, temp)
    results2 = pool.map(extended_multi_func, extended_features)
    clfs = display_sorted_results(results)
    clfs2 = display_sorted_results(results2)
    return results, results2, clfs, clfs2