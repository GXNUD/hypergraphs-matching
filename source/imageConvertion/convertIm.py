from __future__ import division
import time
import math
from PIL import Image, ImageFilter
from numpy import array, zeros, sqrt, int_, asscalar
import numpy as np
from matplotlib.pyplot import imshow, show, subplot, gray, axis, title

'''
Input : Image, Base point(tupla (x, y)) , parameter Beta
Output: A list with neighbors(tuples)
'''
def getNeighborhood(im, point, beta):
    # print "Getting neighborhood"
    Nbeta = [] # neighbors of point(x, y) which distance is less or equal to beta
    N = 0 # number of neighbors
    x, y = point[0], point[1]
    for i in xrange(x - beta, x + beta + 1):
        for j in xrange(y - beta, y + beta + 1):
            if (i >= 0 and j >= 0 and i < im.shape[0] and j < im.shape[1]):
                Nbeta.append((i, j))
                N += 1
    # Debug
    # print "Vecinos %d" % N
    return Nbeta

'''
Input : Image, a list with points(tuples) which belong to the neighborhood
Output: A float number(standard deviation), This is the parameter alpha
'''
def getAlpha(im, neighborhood):
    # print "Getting alpha"
    N = 0
    sumNeighbors = 0
    sumDesv = 0
    for p in neighborhood:
        px = p[0]
        py = p[1]
        sumNeighbors += asscalar(im[px][py])
        N += 1

    average = sumNeighbors / N
    for p in neighborhood:
        px = p[0]
        py = p[1]
        sumDesv += (asscalar(im[px][py]) - average) * (asscalar(im[px][py]) - average)

    alpha = sqrt( sumDesv / (N - 1) )
    # Debug
    # print "Numero de vecinos %d" % (N)
    # print "Media %d" % (average)
    # print "Desv %f" % (alpha)
    return alpha

'''
Put here whatever distance function which involves two points and their feactures
input: Image, Base point(tuple), other point(tuple)
Output: value of distance
'''
def distance(im, basePoint, otherPoint):
    ax, ay = basePoint[0], basePoint[1]
    bx, by = otherPoint[0], otherPoint[1]
    return abs(asscalar(im[ax][ay]) - asscalar(im[bx][by]))

def getHyperEdge(im, neighborhood, alpha, basePoint):
    # print "Getting hyperEdge"
    hyperEdge = []
    for neighbor in neighborhood:
        if (distance(im, basePoint, neighbor) <= alpha):
            hyperEdge.append(neighbor)

    return hyperEdge

'''
So far this calculates An HyperEdge for each pixel
Input: Black & White Image, Beta parameter
Output: Fill HyperEdges matrix and X vector(set of points)
'''
def getHypergraph(im, beta):
    print "Running ..."
    X = []
    E = [[None] * im.shape[1]] * im.shape[0] # HyperEdges of each pixel
    for i in xrange(im.shape[0]):
        for j in xrange(im.shape[1]):
            currentPoint = (i, j)
            neighborhood = getNeighborhood(im, currentPoint, beta)
            alpha = getAlpha(im, neighborhood)
            E[i][j] = getHyperEdge(im, neighborhood, alpha, currentPoint)
            X.append(currentPoint)

    # ready for Hypergraph creation

    # Debug
    print "Finished:"
    for p in X:
        px, py = p[0], p[1]
        print "Point (%d, %d) has %d nodes in his hyperEdge" % (px, py, len(E[px][py]))

if __name__ == '__main__':
    from sys import argv
    if len(argv) < 2:
        print "Usage: python %s <image>" % argv[0]
        exit()
    im = array(Image.open(argv[1]))
    im = im[:,:,0]
    gray()

    start_time = time.time();
    getHypergraph(im, 3)
    print "Execution time %s seconds" % (time.time() - start_time);

    subplot(1, 2, 1)
    imshow(im)
    axis('off')
    title('Image Tested')

    # subplot(1, 2, 2)
    # imshow(imBW)
    # axis('off')
    # title('Test')

    # show()
