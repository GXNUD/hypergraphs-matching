import math
import cv2
from random import random, randint
import numpy as np
from numpy import linalg as LA
np.set_printoptions(precision=3)

def pow_2(base):
	return math.pow(base, 2)

################################################################################

def __compareHist(hist1, hist2):
	n = len(hist1)
	similarity = math.sqrt(sum(pow_2(hist1[i] - hist2[i]) for i in xrange(n)))
	return similarity

def getDistances(points): # Euclidean distances
	n = len(points)
	dist = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			dist[i][j] = math.sqrt(pow_2(points[j].pt[0] - points[i].pt[0]) + pow_2(points[j].pt[1] - points[i].pt[1]))
	return dist

def getDistancesDes(descriptors):
	n = len(descriptors)
	dist = np.zeros((n, n))
	for i in xrange(n):
		for j in xrange(i, n):
			dist[i][j] = dist[j][i] = __compareHist(descriptors[i], descriptors[j])
	return dist

################################################################################

def getFeatures(img, limit= 10, outname="sample", show=False):
	'''
	img should be gray
	'''
	detector = cv2.FeatureDetector_create('SIFT')
	descriptor = cv2.DescriptorExtractor_create('SIFT')
	kp = detector.detect(img)
	kp = sorted(kp, key=lambda x:x.response, reverse=True) # getting most relevant points
	kp, des = descriptor.compute(img, kp)
	if show:
		img_to_write = np.zeros(img.shape)
		img_to_write = cv2.drawKeypoints(img, kp[:limit], img_to_write)
		cv2.imwrite(outname, img_to_write)
		cv2.namedWindow('Keypoints', cv2.WINDOW_NORMAL)
		cv2.imshow('Keypoints', img_to_write)
		cv2.waitKey()
	print len(kp)
	return (kp[:limit], des[:limit]) if len(kp) > limit else (kp, des)

################################################################################

def getMatrixH(distIm1, distIm2, gamma=2):
	# outfile = open("salidaH", "w")
	n, m = len(distIm1), len(distIm2)
	dim = n * m
	H = np.zeros((dim, dim))
	# mm = [[""] * dim] * dim
	hi, hj = 0, 0
	# cv2.waitKey()
	for a in xrange(m):
		for i in xrange(n):
			for b in xrange(m):
				for j in xrange(n):
					H[hi][hj] = math.exp(-gamma * pow_2(abs(distIm1[i][j] - distIm2[a][b])))
					# print ("H[%d][%d], H[(%d, %d)][(%d, %d)]" % (hi+1, hj+1, i+1, a+1, j+1, b+1))
					# print "IM1: %f, IM2: %f\n" % (distIm1[i][j], distIm2[a][b])
					# mm[hi][hj] = ("H[(%d, %d)][(%d, %d)]" % (i+1, a+1, j+1, b+1))
					# outfile.write(str(mm[hi][hj]) + ' | ')
					hj = hj + 1
					if (hj == (n * m)):
						hj = 0
						hi = hi+1
						# outfile.write('\n')
	# outfile.close()
	return H

def getEigenvector(H):
	N = H.shape[0]
	v = np.asarray([randint(1,100) for _ in xrange(N)])
	x = 0
	while x < 24:
		ans = H.dot(v)
		m = sum(pow_2(ans[i]) for i in xrange(ans.shape[0]))
		v = (1/math.sqrt(m)) * (ans)
		x += 1
	return v

def discretize(v, row, col):
	taken = 0
	c = []
	pairs = []
	matches = []
	for i in xrange(col):
		for j in xrange(row):
			pairs.append((j, i))
	while (taken < v.size):
		maxi = v.argmax()
		c.append(pairs[maxi])
		taken += 1
		matches.append(cv2.DMatch(pairs[maxi][0], pairs[maxi][1], v[maxi]))
		v[maxi] = -1
		for index, pair in enumerate(pairs):
			if v[index] != -1:
				if (pair[0] == pairs[maxi][0] or pair[1] == pairs[maxi][1]): # ono to one constraint
					taken += 1
					v[index] = -1
	return matches

################################################################################

def drawMatches(img1, kp1, img2, kp2, matches, des1, des2):
	(rows1, cols1) = img1.shape
	(rows2, cols2) = img2.shape
	out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
	out[:rows1, :cols1] = np.dstack([img1, img1, img1])
	out[:rows2, cols1:] = np.dstack([img2, img2, img2])
	for mat in matches:
		__queryIdx = mat.queryIdx
		__trainIdx = mat.trainIdx

		(x1, y1) = kp1[__queryIdx].pt
		(x2, y2) = kp2[__trainIdx].pt
		print (x1, y1), "  matchs with ", (x2, y2), " value ", mat.distance
		print "Descriptor 1:\n", des1[__queryIdx]
		print "Descriptor 2:\n", des2[__trainIdx]
		cv2.circle(out, (int(x1), int(y1)), 8, (0, 0, 255), 1)
		cv2.circle(out, (int(x2) + cols1, int(y2)), 8, (0, 255, 0), 1)
		cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (102, 255, 255), 1)
	cv2.namedWindow('Matching', cv2.WINDOW_NORMAL)
	cv2.imshow("Matching", out)
	cv2.waitKey()
	cv2.destroyWindow("Matching")
	return out

################################################################################

def main():
	img1 = cv2.imread('./house/house.seq80.png')
	img2 = cv2.imread('./house/house.seq85.png')
	# convert to gray
	img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	# get features and distances
	(kpts1, des1) = getFeatures(img1_gray, 25, './images/original_keypoints.jpg', show=True)
	(kpts2, des2) = getFeatures(img2_gray, 60, './images/model_keypoints.jpg', show=True)

	dist1 = getDistancesDes(des1)
	dist2 = getDistancesDes(des2)


	H = getMatrixH(dist1, dist2)
	eig = getEigenvector(H)
	# show images
	# cv2.imshow("Original", img1)
	# cv2.imshow("Model", img2)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# get matches
	matches = discretize(eig, len(kpts1), len(kpts2))
	result = drawMatches(img1_gray, kpts1, img2_gray, kpts2, matches, des1, des2)

if __name__ == '__main__':
	main()
