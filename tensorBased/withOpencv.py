import math
import cv2
import numpy as np


def pot_2(base):
	return base * base

def getDistances(points):
	dist = np.zeros((len(points),len(points)))
	for i in range(0, dist.shape[0]):
		for j in range(0, dist.shape[1]):
			dist[i][j] = math.sqrt( pot_2(points[j][0] - points[i][0]) + pot_2(points[j][1] - points[i][1]))
	return dist

def getPoints(img, outname):
	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	kp,des = sift.detectAndCompute(imgGray, None)
	# outimg = cv2.drawKeypoints(imgGray, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	# cv2.imwrite(outname, outimg)
	points = []
	# print (des)
	for kpoint in kp:
		x = kpoint.pt[0]
		y = kpoint.pt[1]
		points.append((x, y))
	return points


def getMatrixH(distIm1, distIm2, gamma=2):
	#print len(distIm1), len(distIm2)

	H = np.zeros((len(distIm1) * len(distIm2), len(distIm1) * len(distIm2)))
	#print H.shape[0], H.shape[1]
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

def main():
	img = cv2.imread('gato.jpg')
	(h, w) = img.shape[:2]
	center = (w / 2, h / 2)

	# rotate the image by 180 degrees
	M = cv2.getRotationMatrix2D(center, 10, 1.0)
	rotated = cv2.warpAffine(img, M, (w, h))
	points = getPoints(img, "originalPoints.jpg")
	trans = getPoints(rotated, "rotatedPoints.jpg")
	dist1 = getDistances(points)
	dist2 = getDistances(trans)

	#print (dist1)
	#Print (dist2)
	H = getMatrixH(dist1, dist2)
	print (H)
	# show images
	# cv2.imshow("Original", img)
	# cv2.imshow("Rotated", rotated)
	# cv2.waitKey(0)

if __name__ == '__main__':
	main()
