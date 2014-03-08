clf = NeuralNet([("sig", 250), ("sig", 250)])
print test_clf(FEATURES[0], GAP, clf, num=1)

layers = []
from itertools import product
vals = [0, 10, 50, 100, 200, 400]
possible_layers = set(tuple(x for x in vals if x) for vals in product(vals, vals, vals, vals))
print len(possible_layers)

def func(layers):
    print layers
    new = zip(["sig"]*len(layers), layers)
    clf = NeuralNet(new)
    res = test_clf(FEATURES[0], GAP, clf, num=1)
    print res
    return res

def func(num): return test_clf(FEATURES[0], GAP, NeuralNet([("sig", x)]), num=1)[1][0]


pool = multiprocessing.Pool(processes=4)
results = pool.map(func, possible_layers)

for i, layers in enumerate(possible_layers):
    new = zip(["sig"]*len(layers), layers)
    clf = NeuralNet(new)
    print i, layers,
    print test_clf(FEATURES[0], GAP, clf, num=1)