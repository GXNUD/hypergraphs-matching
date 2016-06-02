import math
import cv2
from random import random, randint
import numpy as np

def pot_2(base):
	return base * base

def getDistances(points):
	dist = np.zeros((len(points) % 100, len(points) % 100))
	for i in range(0, dist.shape[0]):
		for j in range(0, dist.shape[1]):
			dist[i][j] = math.sqrt( pot_2(points[j][0] - points[i][0]) + pot_2(points[j][1] - points[i][1]))
	return dist

def getPoints(img, outname):
	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	kp = sift.detect(imgGray,None)
	img = cv2.drawKeypoints(imgGray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv2.imwrite(outname, img)
	points = []
	print (len(kp))
	for kpoint in kp:
		x = kpoint.pt[0]
		y = kpoint.pt[1]
		points.append((x, y))
	return points


def getMatrixH(distIm1, distIm2, gamma=2):
	H = np.zeros((len(distIm1) * len(distIm2), len(distIm1) * len(distIm2)))
	hi = 0
	hj = 0
	for i in range(0, len(distIm1)-1):
		for a in range(0, len(distIm2)-1):
			for b in range(0, len(distIm2)-1):
				for j in range(0, len(distIm1)-1):
					H[hi][hj] = math.exp(-gamma * pot_2(abs(distIm1[i][j] - distIm2[a][b])))
					hj = hj+1
					if(hj > ((len(distIm1) * len(distIm2)) - 1)):
						hj = 1
						hi = hi+1
	return H

def getEigenvalue(H):
	v = [randint(0,100) for _ in range(H.shape[0])]
	v = np.asarray(v)
	x = 0
	print ("H shape: ")
	print (H.shape)
	while x < 24:
		m = 0
		v = H.dot(v)
		# print ("V before: ")
		# print  (v)
		# print (v.shape)
		for i in range(v.shape[0]):
			m += pot_2(v[i])
		# print ("M: ")
		# print (math.sqrt(m))
		v= (1/math.sqrt(m))*(v)
		x += 1
	# print ("after")
	# print (v)}
	count = 0
	for i in range(v.shape[0]):
		if v[i] > 0:
			count += 1
			print (v[i])
	print ("total: ")
	print (count)

def main():
	img = cv2.imread('paisaje.jpg')
	(h, w) = img.shape[:2]
	center = (w / 2, h / 2)

	# rotate the image by 180 degrees
	M = cv2.getRotationMatrix2D(center, 90, 1.0)
	rotated = cv2.warpAffine(img, M, (w, h))
	points = getPoints(img, 'original_keypoints.jpg')
	trans = getPoints(rotated, 'rotated_keypoints.jpg')
	dist1 = getDistances(points)
	dist2 = getDistances(trans)

	#print (dist1)
	#Print (dist2)
	H = getMatrixH(dist1, dist2)
	# print (H)
	eg = getEigenvalue(H)

	# show images
	# cv2.imshow("Original", img)
	# cv2.imshow("Rotated", rotated)
	# cv2.waitKey(0)

if __name__ == '__main__':
	main()
