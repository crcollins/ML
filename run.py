import csv
import numpy

data = []
with open("data_clean.csv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for id, path, name, exact, feat, opts, occ, virt, orb, dip, eng, gap, time in reader:
        if id == '1' and feat != '[]':
            data.append([exact, numpy.matrix(feat), float(occ), float(virt), int(orb), float(dip), float(eng), float(gap)])

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

def get_learning_curves(X, y, step=25):
    M, N = X.shape
    for lim in xrange(50, M, step):
        W, e1, e2 = get_weight(X, y, limit=lim)
        print lim, e1, e2

print "Gap"
get_learning_curves(X, GAP)
print "HOMO"
get_learning_curves(X, HOMO)
print "LUMO"
get_learning_curves(X, LUMO)
# WH = get_weight(X, HOMO)
# WL = get_weight(X, LUMO)
# WD = get_weight(X, DIPOLE)
# WE = get_weight(X, ENERGY)

# f = numpy.matrix(numpy.concatenate([LUMO-HOMO, numpy.ones(HOMO.shape)],1))
# Wf = get_weight(f, GAP)
# f2 = numpy.matrix(numpy.concatenate([X*WL-X*WH, numpy.ones(HOMO.shape)],1))
# Wf = get_weight(f2, GAP)



