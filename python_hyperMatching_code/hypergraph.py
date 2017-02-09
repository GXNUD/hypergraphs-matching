import matplotlib.tri as tri
import itertools
from math import sqrt, pow


class Hypergraph():
    def __init__(self, kpts, desc):
        self.kpts = kpts
        self.desc = desc
        self.size = len(kpts)
        self.V = [i for i in xrange(len(kpts))]
        self.E = self.__getHyperedges(kpts)
        # self.E = self.__getBfHyperedges()

    def __getHyperedges(self, kpts):
        '''
        builds the hyperedges using the 2 nearest neighbors of each point
        '''
        x, y = zip(*[k.pt for k in kpts])
        triangulation = tri.Triangulation(x, y)

        return triangulation.get_masked_triangles()

    def __getBfHyperedges(self):
        return list(itertools.combinations(range(self.size), 3))

    def __euclideanDistance(self, kpt1, kpt2):
        return sqrt(
            pow(kpt2.pt[0] - kpt1.pt[0], 2) + pow(kpt2.pt[1] - kpt1.pt[1], 2)
        )

    def __getMinValues(self, vec, base):
        '''
        base is the index of the node we want to
        get the 2 minimum values in vec
        '''
        min1, min2 = -1, -1
        minValue = 1e6
        for i, v in enumerate(vec):
            if v < minValue and i != base:
                min1 = i
                minValue = v
        minValue = 1e6
        for i, v in enumerate(vec):
            if v < minValue and i != base and i != min1:
                min2 = i
                minValue = v
        return min1, min2

    def show_hyperedges(self):
        for i, e in enumerate(self.E):
            print "E{0}: {1}".format(i, e)
