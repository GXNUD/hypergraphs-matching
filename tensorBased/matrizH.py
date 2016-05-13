from __future__ import division
from random import random
from numpy import square, power, zeros
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
		trans.append((imagen[x][1],imagen[x][0]));
	# puntos adicionales
	for k in xrange(0, 2):
		nx, ny = generatePoint(5,100,5,100)
		trans.append((nx, ny))
	return trans

def showVector(vec):
	for i in vec:
		print i

def converToVector(matriz):
	pass

def getH(distIm1, distIm2, gamma=2):
	H = zeros((pot_2(len(distIm1)), pot_2(len(distIm2))))
	print H.shape[0], H.shape[1]
	i_H = j_H = 0
	for i_H in xrange(0, H.shape[0]):
		for j_H in xrange(0, H.shape[1]):
			for i in xrange(0, distIm1.shape[0]):
				for j in xrange(0, distIm1.shape[1]):
					for h in xrange(0, distIm2.shape[0]):
						for k in xrange(0, distIm2.shape[1]):
							print "dis1 ", distIm1[i][j]
							print "dis2 ", distIm2[h][k]
							print "exp ", math.exp(-gamma * pot_2(abs(distIm1[i][j] - distIm2[h][k])))
							H[i_H][j_H] = math.exp(-gamma * pot_2(abs(distIm1[i][j] - distIm2[h][k])))
	return H

imagen = generatePoints(5)
transpuesta = transformPoints(imagen)
dist = getDistances(imagen)
dist2 = getDistances(transpuesta)

#for i in xrange(1,len(imagen)):
#	for j in xrange(1,len(imagen)):
#		print "Dist entre", imagen[i], " y ", imagen[j]
#		print dist[i][j]

H = getH(dist, dist2)
for i in xrange(0, H.shape[0]):
	for j in xrange(0, H.shape[1]):
		print "%f " % (H[i][j])
	print '\n'
