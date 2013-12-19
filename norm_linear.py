import random

import numpy

from data import homo, lumo, energy, dipole, gap

def normalize(X):
    mu = X.mean(0)
    sigma = X.std(0)
    return (X-mu)/sigma, mu, sigma


offset = 200
order = range(len(homo))
random.shuffle(order)
homo = homo[order]
lumo = lumo[order]
energy = energy[order]
dipole = dipole[order]
gap = gap[order]

X = numpy.concatenate((homo[:offset], lumo[:offset], energy[:offset], dipole[:offset]), 1)
X, mu, sigma = normalize(X)
X = numpy.concatenate((X, numpy.ones(homo[:offset].shape)), 1)
theta = (X.T*X).I*X.T*gap[:offset]
print theta
X2 = numpy.concatenate((homo[offset:], lumo[offset:], energy[offset:], dipole[offset:]), 1)
X2 = (X2-mu)/sigma
estimate = numpy.concatenate((X2, numpy.ones(homo[offset:].shape)),1)*theta
temp = (gap[offset:] - estimate)
print (temp*temp.T).mean()
