import csv
import multiprocessing
import ast
import cPickle

import numpy
from sklearn import decomposition
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from sklearn import dummy
from sklearn import neighbors

from display import display_sorted_results, plot_scan_2d, plot_scan, PCA_stuff
from display import plot_num_samples
from utils import OptimizedCLF


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

FEATURES = []
for group in zip(*tuple(features)):
    FEATURES.append(numpy.matrix(group))


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


def multi_func(xset):
    temp = []
    for yset in (HOMO, LUMO, GAP):
        temp.append(test_sklearn(xset, yset))
    return temp


def multi_func2(params):
    yset, xset = params
    return [test_sklearn(xset, yset)]


def main():
    pool = multiprocessing.Pool(processes=4)
    results = pool.map(multi_func, FEATURES)
    results2 = pool.map(multi_func2, FEATURES1)
    clfs = display_sorted_results(results)
    clfs2 = display_sorted_results(results2)
    return results, results2, clfs, clfs2


def get_test_features(homo, lumo, gap, features):
    try:
        with open("features1.pkl", "rb") as f:
            FEATURES1 = cPickle.load(f)
        if len(FEATURES1) == 4 * len(features):
            return FEATURES1
    except IOError:
        pass

    FEATURES1 = []
    for i, feat in enumerate(features):
        homoclf = OptimizedCLF(feat, homo, svm.SVR, {"C": 10, "gamma": 0.05}).get_optimized_clf()
        homoclf.fit(feat, HOMO.T.tolist()[0])
        HOMOp = numpy.matrix(homoclf.predict(feat)).T

        lumoclf = OptimizedCLF(feat, lumo, svm.SVR, {"C": 10, "gamma": 0.05}).get_optimized_clf()
        lumoclf.fit(feat, LUMO.T.tolist()[0])
        LUMOp = numpy.matrix(lumoclf.predict(feat)).T

        gapclf = OptimizedCLF(feat, gap, svm.SVR, {"C": 10, "gamma": 0.05}).get_optimized_clf()
        gapclf.fit(feat, GAP.T.tolist()[0])
        GAPp = numpy.matrix(gapclf.predict(feat)).T

        FEATURES1.append((GAP, numpy.concatenate([feat, HOMOp, LUMOp], 1)))
        FEATURES1.append((HOMO, numpy.concatenate([feat, LUMOp, GAPp], 1)))
        FEATURES1.append((LUMO, numpy.concatenate([feat, GAPp, HOMOp], 1)))
        FEATURES1.append((GAP, numpy.concatenate([HOMOp, LUMOp, numpy.ones(GAPp.shape)], 1)))
    with open("features1.pkl", "wb") as f:
        cPickle.dump(FEATURES1, f, protocol=-1)
    return FEATURES1

FEATURES1 = get_test_features(HOMO, LUMO, GAP, FEATURES[1:])

results, results2, clfs, clfs2 = main()

if __name__ == "__main__":
    print time.time() - start
    plot_scan_2d(FEATURES[1], GAP, svm.SVR, {"C": [0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000], "gamma": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]})
    plot_scan(FEATURES[1], GAP, linear_model.Ridge, {"alpha": [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]})
    PCA_stuff(FEATURES[1], GAP)
    plot_num_samples(FEATURES[1], GAP, svm.SVR(C=10, gamma=0.05))
