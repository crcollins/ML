import csv
import itertools

import numpy

data = []
with open("data_clean.csv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for id, path, name, exact, feat, feat2, feat3, feat4, feat5, opts, occ, virt, orb, dip, eng, gap, time in reader:
        try:
            data.append([exact, numpy.matrix(feat), numpy.matrix(feat2), numpy.matrix(feat3), numpy.matrix(feat4), numpy.matrix(feat5), float(occ), float(virt), int(orb), float(dip), float(eng), float(gap)])
        except:
            pass

M = len(data)
N = data[0][1].shape[1]
N2 = data[0][2].shape[1]

FEATURES = numpy.zeros((M, N))
FEATURES2 = numpy.zeros((M, N2))
FEATURES3 = numpy.zeros((M, N2))
FEATURES4 = numpy.zeros((M, N2))
FEATURES5 = numpy.zeros((M, N2))
HOMO = numpy.zeros((M, 1))
LUMO = numpy.zeros((M, 1))
DIPOLE = numpy.zeros((M, 1))
ENERGY = numpy.zeros((M, 1))
GAP = numpy.zeros((M, 1))

for i, (name, feat, feat2, feat3, feat4, feat5, occ, virt, orb, dip, eng, gap) in enumerate(data):
    FEATURES[i,:] = feat
    FEATURES2[i,:] = feat2
    FEATURES3[i,:] = feat3
    FEATURES4[i,:] = feat4
    FEATURES5[i,:] = feat5
    HOMO[i] = occ
    LUMO[i] = virt
    DIPOLE[i] = dip
    ENERGY[i] = eng
    GAP[i] = gap
X = numpy.matrix(FEATURES)
X2 = numpy.matrix(FEATURES2)
X3 = numpy.matrix(FEATURES3)
X4 = numpy.matrix(FEATURES4)
X5 = numpy.matrix(FEATURES5)


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



from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from sklearn import dummy
from sklearn import neighbors

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
    return train.mean(), cross.mean()

def test_sklearn(X, y):
    funcs = {
        "dummy": dummy.DummyRegressor(),
        "linear": linear_model.LinearRegression(),
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
        train_results[idx] = train
        test_results[idx] = test
    return train_results, test_results


def sort_xset(xset):
    temp = {}
    for yset in xset:
        for name in yset:
            zipped = zip(*yset[name])
            (_, test) = zipped[0]
            if temp.get(name):
                temp[name].append(test)
            else:
                temp[name] = [test]
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
                        lines[i].extend(['', train, test])
                    except IndexError:
                        lines.append(['', train, test])
            for line in lines:
                tests = line[2::3]
                line.extend(['', sum(tests)/len(tests)])
                print ','.join(str(x) for x in line)
        print '\n'


results = []
for xset in (X, X2, X3, X4, X5):
    temp = []
    for yset in (HOMO, LUMO, GAP):
        temp.append(test_sklearn(xset, yset))
    results.append(temp)



display_sorted_results(results)

# clf = svm.SVR(C=20, kernel="rbf", gamma=0.1)
# clf.fit(AA["learn"]["X"], AA["learn"]["GAP"].T.tolist()[0])
# temp = numpy.matrix(clf.predict(AA["learn"]["X"])).T
# print numpy.abs(temp-AA["learn"]["GAP"]).mean()
# print numpy.abs(temp.T-AA["learn"]["GAP"]).mean()

# temp = numpy.matrix(clf.predict(AA["test"]["X"])).T
# print numpy.abs(temp-AA["test"]["GAP"]).mean()
# print numpy.abs(temp.T-AA["test"]["GAP"]).mean()

# print "Gap"
# get_learning_curves(X, GAP)
# print "HOMO"
# get_learning_curves(X, HOMO)
# print "LUMO"
# get_learning_curves(X, LUMO)
# WH = get_weight(X, HOMO)
# WL = get_weight(X, LUMO)
# WD = get_weight(X, DIPOLE)
# WE = get_weight(X, ENERGY)

# f = numpy.matrix(numpy.concatenate([LUMO-HOMO, numpy.ones(HOMO.shape)],1))
# Wf = get_weight(f, GAP)
# f2 = numpy.matrix(numpy.concatenate([X*WL-X*WH, numpy.ones(HOMO.shape)],1))
# Wf = get_weight(f2, GAP)


