import itertools

import numpy
import scipy.optimize
from scipy.spatial.distance import cdist
from sklearn import cross_validation
from sklearn.metrics import mean_absolute_error


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
