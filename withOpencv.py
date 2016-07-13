import math
import cv2
from random import random, randint
import numpy as np

def pow_2(base):
	return math.pow(base, 2)

def getDistances(points):
	dist = np.zeros((len(points), len(points)))
	for i in range(dist.shape[0]):
		for j in range(dist.shape[1]):
			dist[i][j] = math.sqrt( pow_2(points[j].pt[0] - points[i].pt[0]) + pow_2(points[j].pt[1] - points[i].pt[1]))
	return dist

def getFeatures(img, outname, show=False):
	'''
	img should be gray
	'''
	limit = 100
	detector = cv2.FeatureDetector_create('SIFT')
	descriptor = cv2.DescriptorExtractor_create('SIFT')
	kp = detector.detect(img)
	kp = sorted(kp, key=lambda x:x.response)
	kp, des = descriptor.compute(img, kp)
	img_to_write = np.zeros(img.shape)
	img_to_write = cv2.drawKeypoints(img, kp, img_to_write)
	cv2.imwrite(outname, img_to_write)
	if show:
		cv2.imshow("Keypoints", img_to_write)
		cv2.waitKey()
	return kp[:limit], des[:limit]


def getMatrixH(distIm1, distIm2, gamma=2):
	# outfile = open("salidaH", "w")
	dim = len(distIm1) * len(distIm2)
	H = np.zeros((dim, dim))
	mm = [[""] * dim] * dim
	hi, hj = 0, 0
	cv2.waitKey()
	for a in range(len(distIm2)):
		for i in range(len(distIm1)):
			for b in range(len(distIm2)):
				for j in range(len(distIm1)):
					H[hi][hj] = math.exp(-gamma * pow_2(abs(distIm1[i][j] - distIm2[a][b])))
					# print ("H[%d][%d], H[(%d, %d)][(%d, %d)]" % (hi+1, hj+1, i+1, a+1, j+1, b+1))
					# print "IM1: %f, IM2: %f\n" % (distIm1[i][j], distIm2[a][b])
					# mm[hi][hj] = ("H[(%d, %d)][(%d, %d)]" % (i+1, a+1, j+1, b+1))
					# outfile.write(str(mm[hi][hj]) + ' | ')
					hj = hj + 1
					if (hj == (len(distIm1) * len(distIm2))):
						hj = 0
						hi = hi+1
						# outfile.write('\n')
	# outfile.close()
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

def discretize(v, row, col):
	taken = 0
	pairs = [] # L
	c = []
	matches = []
	for i in range(row):
		for j in range(col):
			pairs.append((j, i))
	# print "init v ", v
	# print (pairs)
	while (taken <= v.size):
		maxi = v.argmax()
		c.append(pairs[maxi])
		matches.append(cv2.DMatch(pairs[maxi][0], pairs[maxi][1], v[maxi]))
		# print "taken pair", pairs[maxi], " with value ", v[maxi]
		taken += 1
		for index, p in enumerate(pairs):
			if (p[0] == pairs[maxi][0] or p[1] == pairs[maxi][1]) and p != pairs[maxi]: # ono to one
				# print "has same", p, pairs[maxi]
				taken += 1
				v[index] = -1
		v[maxi] = -1
	return matches

def drawMatches(img1, kp1, img2, kp2, matches):
	print img1.shape
	print img2.shape
	(rows1, cols1) = img1.shape
	(rows2, cols2) = img2.shape

	out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

	out[:rows1, :cols1] = np.dstack([img1, img1, img1])
	out[:rows2, cols1:] = np.dstack([img2, img2, img2])
	for mat in matches:
		img1_idx = mat.queryIdx
		img2_idx = mat.trainIdx

		(x1, y1) = kp1[img1_idx].pt
		(x2, y2) = kp2[img2_idx].pt

		cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
		cv2.circle(out, (int(x2) + cols1, int(y2)), 4, (255, 0, 0), 1)

		cv2.line(out, (int(x1), int(y1)), (int(x1) + cols1, int(y2)), (255, 0, 0), 1)
	cv2.imshow("Matching", out)
	cv2.waitKey()
	cv2.destroyWindow("Matching")
	return out

def match(des1, des2):
	# ge distances with l2 norm
	pass

def main():
	img1 = cv2.imread('./house/house.seq29.png')
	img2 = cv2.imread('./house/house.seq35.png')
	# rotate the image by 180 degrees
	# (h, w) = img1.shape[:2]
	# center = (w / 2, h / 2)
	# M = cv2.getRotationMatrix2D(center, 90, 1.0)
	# img2 = cv2.warpAffine(img1, M, (w, h))

	# convert to gray
	img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	# get features and distances
	(kpts1, des1) = getFeatures(img1_gray, 'original_keypoints.jpg', show=False)
	(kpts2, des2) = getFeatures(img2_gray, 'model_keypoints.jpg', show=False)

	dist1 = getDistances(kpts1)
	dist2 = getDistances(kpts2)

	H = getMatrixH(dist1, dist2)
	eig = getEigenvector1(H)
	# show images
	# cv2.imshow("Original", img1)
	# cv2.imshow("Model", img2)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# get matches
	matches = discretize(eig, len(kpts1), len(kpts2))
	result = drawMatches(img1_gray, kpts1, img2_gray, kpts2, matches)

if __name__ == '__main__':
	main()
