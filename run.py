import csv

import numpy

data = []
with open("data_clean.csv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for id, path, name, exact, feat, opts, occ, virt, orb, dip, eng, gap, time in reader:
        try:
            data.append([exact, numpy.matrix(feat), float(occ), float(virt), int(orb), float(dip), float(eng), float(gap)])
        except:
            pass

M = len(data)
N = data[0][1].shape[1]

FEATURES = numpy.zeros((M, N))
HOMO = numpy.zeros((M, 1))
LUMO = numpy.zeros((M, 1))
DIPOLE = numpy.zeros((M, 1))
ENERGY = numpy.zeros((M, 1))
GAP = numpy.zeros((M, 1))

for i, (name, feat, occ, virt, orb, dip, eng, gap) in enumerate(data):
    FEATURES[i,:] = feat
    HOMO[i] = occ
    LUMO[i] = virt
    DIPOLE[i] = dip
    ENERGY[i] = eng
    GAP[i] = gap
X = numpy.matrix(FEATURES)




def get_weight(X, y, limit=400):
    Xin = X[:limit,  :]
    Xout = X[limit:, :]
    yin = y[:limit,  :]
    yout = y[limit:, :]
    W = numpy.linalg.pinv(Xin.T*Xin)*Xin.T*yin
    error = numpy.abs(yin-Xin*W).mean()
    cross_error = numpy.abs(yout-Xout*W).mean()
    return W, error, cross_error

def test_it(clf, X, y, limit=400):
    Xin = X[:limit,  :]
    Xout = X[limit:, :]

    yin = y[:limit,  :]
    yout = y[limit:, :]
    ylist = yin.T.tolist()[0]

    clf.fit(Xin, ylist)
    temp = numpy.matrix(clf.predict(Xin)).T
    error = numpy.abs(temp-yin).mean()

    temp = numpy.matrix(clf.predict(Xout)).T
    cross_error = numpy.abs(temp-yout).mean()
    return error, cross_error


def get_learning_curves(X, y, step=25):
    M, N = X.shape
    for lim in xrange(50, M, step):
        W, e1, e2 = get_weight(X, y, limit=lim)
        print lim, e1, e2


from sklearn import linear_model
from sklearn import svm
from sklearn import tree

def test_sklearn(X, y):
    funcs = {
        "linear": linear_model.LinearRegression(),
        "linear ridge .05": linear_model.Ridge(alpha = .05),
        "linear ridge .5": linear_model.Ridge(alpha = .5),
        "linear ridge 5": linear_model.Ridge(alpha = 5),
        "LARS .01": linear_model.LassoLars(alpha=.01),
        "LARS .1": linear_model.LassoLars(alpha=.1),
        "LARS 1": linear_model.LassoLars(alpha=1),
        "svm": svm.SVR(),
        "svm rbf": svm.SVR(kernel='rbf'),
        "svm rbf 2": svm.SVR(C=0.1, kernel="rbf", gamma=0.1),
        "svm rbf 3": svm.SVR(C=20, kernel="rbf", gamma=0.1),
        "tree": tree.DecisionTreeRegressor(),
        "tree 1": tree.DecisionTreeRegressor(max_depth=1),
        "tree 10": tree.DecisionTreeRegressor(max_depth=10),
        "tree 100": tree.DecisionTreeRegressor(max_depth=100),
    }
    ylist = y.T.tolist()[0]
    M, N = X.shape
    for name, clf in funcs.items():
        print name
        for lim in xrange(50, M, 25):
            e1, e2 = test_it(clf, X, y, limit=lim)
            print lim, e1, e2
        print

test_sklearn(X, GAP)

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



