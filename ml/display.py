import numpy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import dummy
from sklearn import neighbors
from sklearn import cross_validation
from sklearn.metrics import r2_score

from utils import scan, test_clf


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
    return Xs, Ys, COLOR


def plot_PCA_background(Xs, Ys, Zs, method='nearest'):
    from scipy.interpolate import griddata
    xgrid = numpy.linspace(Xs.min(), Xs.max(), 1000)
    ygrid = numpy.linspace(Ys.min(), Ys.max(), 1000)
    XX, YY = numpy.meshgrid(xgrid, ygrid)
    points = numpy.concatenate([numpy.matrix(Xs).T, numpy.matrix(Ys).T], 1)
    grid_z2 = griddata(points, Zs, (XX, YY), method=method)
    plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower')
    # cm = plt.get_cmap("HOT")
    # X = (Xs-Xs.min())/(Xs.max()-Xs.min())
    # Y = (Ys-Ys.min())/(Ys.max()-Ys.min())
    # plt.scatter(X, Y, c=Zs,s=80, marker='o', edgecolors='none')
    plt.show()


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
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(test, interpolation='nearest')
    fig.colorbar(cax)
    ax.xaxis.set_ticks(numpy.arange(0,len(listform[0][1])))
    ax.set_xticklabels(listform[0][1])
    ax.yaxis.set_ticks(numpy.arange(0,len(listform[1][1])))
    ax.set_yticklabels(listform[1][1])
    plt.xlabel(listform[0][0])
    plt.ylabel(listform[1][0])
    plt.show()
    return train, test


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


def plot_homo_lumo_gap(homo, lumo, gap):
    COLOR = (gap-gap.min())/gap.max()
    cm = plt.get_cmap("HOT")
    plt.scatter(homo, lumo, c=COLOR, s=20, marker='o', edgecolors='none')
    plt.xlabel("HOMO (eV)")
    plt.ylabel("LUMO (eV)")
    plt.show()


def plot_actual_prediction(actual, prediction, label=""):
    lim = actual.max()
    r2 = r2_score(actual, prediction)
    plt.plot(actual, prediction, 'b.')
    plt.plot([0,lim],[0,lim], 'r', label="$r^2=%.2f$"%r2)
    plt.xlabel("%s Actual (eV)" % label)
    plt.ylabel("%s Prediction (eV)" % label)
    plt.title("%s Actual verses Prediction" % label)
    plt.legend(loc="best")
    plt.show()


def plot_scan_2d_surface(X, y, function, params):
    listform = params.items()
    train, test = scan(X, y, function, params)
    surface(test)
    return train, test


def surface(data, colormap="jet"):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = numpy.arange(data.shape[0])
    Y = numpy.arange(data.shape[1])
    XX, YY = numpy.meshgrid(X, Y)
    surf = ax.plot_surface(XX, YY, data, rstride=1, cstride=1, cmap=colormap, linewidth=0, antialiased=False, shade=True)
    ax.set_zlim(data.min(), data.max())
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
