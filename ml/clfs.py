import numpy

from sklearn import svm

from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection

from utils import gauss_kernel_gen, laplace_kernel_gen

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