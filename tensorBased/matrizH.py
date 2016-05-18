from __future__ import division
from random import random, randint
from numpy import square, power, zeros, asarray
import numpy as np
import math

def generatePoint(minX,maxX,minY,maxY):
	px = ((random()*100)%(maxX-minX))+minX;
	py = ((random()*100)%(maxY-minY))+minY;
	return int(px), int(py);

def pot_2(base):
	return base * base

def getDistances(points):
	dist = zeros((len(points), len(points)))

	for i in xrange(0, dist.shape[0]):
		for j in xrange(0, dist.shape[1]):
			dist[i][j] = math.sqrt( pot_2(points[j][0] - points[i][0]) + pot_2(points[j][1] - points[i][1]))
	return dist

def generatePoints(N):
	'''
	returna un vector de tuplas, cada i es un punto
	'''
	points = []
	for x in xrange(0,N):
		px, py = generatePoint(5,100,5,100)
		points.append((px, py));
	return points

def transformPoints(points):
	trans = []
	for x in xrange(0, len(points)):
		trans.append((points[x][1],points[x][0]));
	# puntos adicionales
	# for k in xrange(0, 2):
	# 	nx, ny = generatePoint(5,100,5,100)
	# 	trans.append((nx, ny))
	return trans

def showVector(vec):
	for i in vec:
		print i

def converToVector(matriz):
	pass

def getH(distIm1, distIm2, gamma=2):
	H = zeros((pot_2(len(distIm1)), pot_2(len(distIm2))))
	print H.shape[0], H.shape[1]
	for i in xrange(0, H.shape[0]):
		for j in xrange(0, H.shape[1]):
			ia = i/5
			ja = i%5
			ib = j/5
			jb = j%5
			H[i][j]= math.exp(-gamma * pot_2(abs(distIm1[ia][ja] - distIm2[ib][jb])))
	return H

def getEigenvalue(H):
	v = [randint(0,100) for _ in xrange(H.shape[0])]
	v = asarray(v)
	print "H", H
	print "Before", v
	v = H.dot(v)
	print "After: ", v
	print v
	return v

imagen1 = generatePoints(5)
imagen2 = generatePoints(5)
print "Image 1", imagen1, '\n'
print "Image 2", imagen2, '\n'
transpuesta = transformPoints(imagen1)
dist1 = getDistances(imagen1)
dist2 = getDistances(imagen2)
print "Dist 1", dist1, '\n'
print "Dist 2", dist2, '\n'
#for i in xrange(1,len(imagen)):
#	for j in xrange(1,len(imagen)):
#		print "Dist entre", imagen[i], " y ", imagen[j]
#		print dist[i][j]

H = getH(dist1, dist2)
V = getEigenvalue(H)
# for i in xrange(0, H.shape[0]):
# 	for j in xrange(0, H.shape[1]):
# 		print "%f " % (H[i][j])
# 	print '\n'
