import csv
import ast
import cPickle

import numpy

import ml.ml


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
FEATURES1 = ml.ml.get_extended_features(HOMO, LUMO, GAP, FEATURES[1:])


results, results2, clfs, clfs2 = ml.ml.main((HOMO, ), FEATURES, FEATURES1)

if __name__ == "__main__":
    print time.time() - start
    plot_scan_2d(FEATURES[1], GAP, svm.SVR, {"C": [0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000], "gamma": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]})
    plot_scan(FEATURES[1], GAP, linear_model.Ridge, {"alpha": [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]})
    PCA_stuff(FEATURES[1], GAP)
    plot_num_samples(FEATURES[1], GAP, svm.SVR(C=10, gamma=0.05))