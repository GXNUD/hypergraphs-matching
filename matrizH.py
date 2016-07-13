from __future__ import division
from random import random, randint, shuffle
from numpy import square, power, zeros, asarray
import numpy as np
from numpy import linalg as LA
import math

def generatePoint(minX,maxX,minY,maxY):
	px = ((random()*100)%(maxX-minX))+minX;
	py = ((random()*100)%(maxY-minY))+minY;
	return int(px), int(py)

def generatePoints(N):
	'''
	retorna un vector de tuplas, cada i es un punto
	'''
	points = []
	for x in range(0, N):
		px, py = generatePoint(5, 100, 5, 100)
		points.append((px, py))
	return points

def pow_2(base):
	return math.pow(base, 2)

def getDistances(points):
	dist = zeros((len(points), len(points)))
	for i in range(dist.shape[0]):
		for j in range(dist.shape[1]):
			dist[i][j] = math.sqrt(pow_2(points[j][0] - points[i][0]) + pow_2(points[j][1] - points[i][1]))
	return dist

def transformPoints(points, noise=0, doShuffle=False):
	'''
	noise es el numero de puntos adicionales para complicar el match
	'''
	trans = []
	for x in range(len(points)):
		trans.append((points[x][1],points[x][0]))

	for k in range(noise):
		nx, ny = generatePoint(5, 100, 5, 100)
		trans.append((nx, ny))
	if doShuffle:
		shuffle(trans)
	return trans

def getMatrixH(distIm1, distIm2, gamma=1):
	outfile = open("salidaH", "w")
	dim = len(distIm1) * len(distIm2)
	H = zeros((dim, dim))
	mm = [[""] * dim] * dim
	hi, hj = 0, 0
	for a in range(len(distIm2)):
		for i in range(len(distIm1)):
			for b in range(len(distIm2)):
				for j in range(len(distIm1)):
					H[hi][hj] = math.exp(-gamma * pow_2(abs(distIm1[i][j] - distIm2[a][b])))
					# print ("H[%d][%d], H[(%d, %d)][(%d, %d)]" % (hi+1, hj+1, i+1, a+1, j+1, b+1))
					# print "IM1: %f, IM2: %f\n" % (distIm1[i][j], distIm2[a][b])
					mm[hi][hj] = ("H[(%d, %d)][(%d, %d)]" % (i+1, a+1, j+1, b+1))
					outfile.write(str(mm[hi][hj]) + ' | ')
					hj = hj + 1
					if(hj == (len(distIm1) * len(distIm2))):
						hj = 0
						hi = hi+1
						outfile.write('\n');
	outfile.close()
	return H

def getEigenvector1(H):
	# print ("getEigenvalue (start)")
	v = [randint(0,100) for _ in range(H.shape[0])]
	v = np.asarray(v)
	x = 0
	# print ("Init V: ")
	# print (v)
	while x < 24:
		m = 0
		ans = H.dot(v)
		for i in range(ans.shape[0]):
			m += pow_2(ans[i])
		v = (1/math.sqrt(m)) * (ans)
		x += 1
		# print ("V: ")
		# print (v)
	return (v)

def getEigenvector2(H): # hadamard
	# print ("getEigenvalue (start)")
	v = [randint(0,100) for _ in range(H.shape[0])]
	v = np.asarray(v)
	x = 0
	# print ("Init V: ")
	# print (v)
	while x < 12:
		m = 0
		hada = np.multiply(v, v)
		p = H.dot(hada)
		v = np.multiply(p, v)
		for i in range(v.shape[0]):
			m += abs(v[i])
		v = (1 / m) * (v)
		x += 1
		# print ("m:")
		# print (m)
		# print ("V: ")
		# print (v)
	count = 0
	# print (v)
	# for i in range(v.shape[0]):
	# 	if v[i] > 0.0:
	# 		count += 1
	# 		print ((v[i]))
	# print ("V final 2: ")
	# print (v)
	# print ("getEigenvalue (end)")
	return (v)

def discretize(v, row, col):
	taken = 0
	pairs = [] # L
	c = []
	for i in range(row):
		for j in range(col):
			pairs.append((j, i))
	# print "v ", v
	# print (pairs)
	while (taken <= v.size):
		maxi = v.argmax()
		c.append(pairs[maxi])
		# print "taken pair", pairs[maxi], " with value ", v[maxi]
		taken += 1
		for index, p in enumerate(pairs):
			if (p[0] == pairs[maxi][0] or p[1] == pairs[maxi][1]) and p != pairs[maxi]: # ono to one
				# print "has same", p, pairs[maxi]
				taken += 1
				v[index] = -1
		v[maxi] = -1
		# print "taken ", taken
		# print "v", v
	print (c)

def drawPoints(im1, im2):
	import matplotlib.pyplot as plt
	plt.plot(*zip(*im1), marker='o', color='r', ls='', label='imagen1')
	plt.plot(*zip(*im2), marker='v', color='b', ls='', label='imagen2')
	plt.show()

def main():
	# imagen1 = generatePoints(3)
	imagen1 = [(1, 1), (4, 1), (4, 5)]
	# print ("imagen1: ")
	# print (imagen1)
	# imagen2 = transformPoints(imagen1, 1, doShuffle=True)
	# imagen2 = generatePoints(5)
	imagen2 = [(0, 0), (4, 0), (4, 3), (3, 0)]
	# print ("imagen2: ")
	# print (imagen2
	# drawPoints(imagen1, imagen2)
	dist1 = getDistances(imagen1)
	# print ("dist1: ")
	# print (dist1)
	dist2 = getDistances(imagen2)
	# print ("dist2: ")
	# print (dist2)
	H = getMatrixH(dist1, dist2)
	# print ("H: ")
	# print (H)
	V1 = getEigenvector1(H)
	w, v = LA.eig(H)
	print "Eigenvector from function", V1
	print "Eigenvector from numpy", w
	# print ("V multiply sqrt(n1): ")
	# print (math.sqrt(len(imagen1)) * V1)
	# V2 = getEigenvector2(H)
	# discretize(V1, len(imagen1), len(imagen2))

if __name__ == '__main__':
	main()
