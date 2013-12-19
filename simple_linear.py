import random

import numpy

from data import homo, lumo, energy, dipole, gap

offset = 200
order = range(len(homo))
random.shuffle(order)
homo = homo[order]
lumo = lumo[order]
energy = energy[order]
dipole = dipole[order]
gap = gap[order]

X = numpy.concatenate((homo[:offset], lumo[:offset], energy[:offset], dipole[:offset], numpy.ones(homo[:offset].shape)), 1)
theta = (X.T*X).I*X.T*gap[:offset]
print theta
estimate = numpy.concatenate((homo[offset:], lumo[offset:], energy[offset:], dipole[offset:], numpy.ones(homo[offset:].shape)),1)*theta
temp = (gap[offset:] - estimate)
print (temp*temp.T).mean()