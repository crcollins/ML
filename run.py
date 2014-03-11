import csv
import ast
import cPickle

import numpy
from sklearn import svm
from sklearn import linear_model

import ml.ml
from ml.display import plot_scan_2d, plot_scan, plot_num_samples
from ml.display import PCA_stuff, plot_PCA_background

data = []
with open("cleaned_data.csv", "r") as csvfile:
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
YSETS = (HOMO, LUMO, GAP)

FEATURES = []
for group in zip(*tuple(features)):
    FEATURES.append(numpy.matrix(group))
FEATURES1 = ml.ml.get_extended_features(HOMO, LUMO, GAP, FEATURES[1:])


results, results2, clfs, clfs2 = ml.ml.main(YSETS, FEATURES, FEATURES1)

if __name__ == "__main__":
    # Make some example plots
    2d_scan_params = {
        "C": [0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000],
        "gamma": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
    }
    plot_scan_2d(FEATURES[1], GAP, svm.SVR, 2d_scan_params)
    scan_params = {"alpha": [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]}
    plot_scan(FEATURES[1], GAP, linear_model.Ridge, scan_params)
    Xs, Ys, Zs = PCA_stuff(FEATURES[1], GAP)
    plot_PCA_background(Xs, Ys, Zs)
    plot_num_samples(FEATURES[1], GAP, svm.SVR(C=10, gamma=0.05))