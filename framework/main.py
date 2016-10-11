from math import sqrt, pow, exp
import numpy as np
from numpy import linalg as LA
import cv2

class Hypergraph():
	def __init__(self, kpts, desc):
		self.V = [i for i in xrange(len(kpts))]
		self.E = self.__getHyperedges(kpts)

	def __getHyperedges(self, kpts):
		''' builds the hyperedges using the k nearest neighbors '''
		E = []
		n = len(kpts)
		dist = np.zeros((n, n))
		for i, r in enumerate(kpts):
			for j, l in enumerate(kpts):
				dist[i][j] = self.__euclideanDistance(r, l)
				print "{0} {1}".format(i, j)
				print "dist( ({0}, {1}) ,({2}, {3})) = {4}".format(r.pt[0], r.pt[1], l.pt[0], l.pt[1], dist[i][j])
		for i, values in enumerate(dist):
			minA, minB = self.__getMinValues(values, i)
			E.append((i, minA, minB))
		return E

	def __euclideanDistance(self, kpt1, kpt2):
		return sqrt(pow(kpt2.pt[0] - kpt1.pt[0], 2) + pow(kpt2.pt[1] - kpt1.pt[1], 2))

	def __getMinValues(self, vec, base):
		'''
		base is the index of the node we want to
		get K = 2 minimum values in vec
		'''
		minA, minB = -1, -1
		minValue = 1e6
		for i, v in enumerate(vec):
			if v < minValue and not i == base:
				minA = i
				minValue = v
		minValue = 1e6
		for i, v in enumerate(vec):
			if v < minValue and not i == base and not i == minA:
				minB = i
				minValue = v
		return minA, minB

	def show_hyperedges(self):
		for i, e in enumerate(self.E):
			print "E{0}: {1}".format(i, e)

def get_features(img, limit=10, outname="sample.jpg", show=False):
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
		cv2.namedWindow('Keypoints')#, cv2.WINDOW_NORMAL)
		cv2.imshow('Keypoints', img_to_write)
		cv2.waitKey()
	return (kp[:limit], des[:limit]) if len(kp) > limit else (kp, des)

#=========================== Feature Based Utils ==============================#

def similarity(des1, des2):
	norm1 = LA.norm(des1)
	norm2 = LA.norm(des2)
	similarity =  np.dot(des1, des2) / (norm1 * norm2)
	return similarity

def similarity_descriptors(target_descriptors, ref_descriptors):
	m = len(target_descriptors)
	dist = np.zeros((m, m))
	for i, t in enumerate(target_descriptors):
		for j, r in enumerate(ref_descriptors):
			dist[i][j] = distance(t, r)
	return dist

#=========================== Feature Based Utils ==============================#

def hyperedge_weigth(edge, W, kpts):
	'''
	det([vi - vk, vj - vk]) * sum 1 / sqrt(norm((vi - vk) * norm(vj - vk)))
	'''
	vi = np.asarray(kpts[edge[0]].pt)
	vj = np.asarray(kpts[edge[1]].pt)
	vk = np.asarray(kpts[edge[2]].pt)
	mat = np.array([vi - vk, vj - vk])
	# det = (mat[0][0] * mat[1][1]) - (mat[1][0] * mat[0][1])
	U = LA.det(mat)
	weigth = U * W
	return weigth

def compute_w_parameter(kpts):
	acum = 0
	for i in kpts:
		for j in kpts:
			for k in kpts:
				vi = np.asarray(i.pt)
				vj = np.asarray(j.pt)
				vk = np.asarray(k.pt)
				acum += 1 / sqrt(LA.norm(vi - vk) * LA.norm(vj - vk)) # TODO FIX division
	return acum

def hyperedges_similarity(ei, ej, w1, w2, kpts1, kpts2):
	a = hyperedge_weigth(ei, w1, kpts1)
	b = hyperedge_weigth(ej, w2, kpts2)
	sigma = 0.5 # TODO wich is the best sigma
	S = exp(-( pow(LA.norm(a - b), 2) / sigma )) # TODO does np.norm work correctly?
	return S

def get_Hs(Hgt, Hgr, kpts1, kpts2):
	'''
	This matrix is a relation between each
	hyperedge from Image 1 and each from image 2
	'''
	E1, E2 = Hgt.E, Hgr.E
	n, m = len(E1), len(E2)
	H = np.zeros((n, m))
	w1 = compute_w_parameter(kpts1)
	w2 = compute_w_parameter(kpts2)
	for i, ei in enumerate(E1):
		for j, ej in xrange(E2):
			H[i][j] = hyperedges_similarity(ei, ej, w1, w2, kpts1, kpts2)
	return H

if __name__ == "__main__":
	M = 5
	img1 = cv2.imread('./house.seq0.png')
	img2 = cv2.imread('./house.seq0.png')
	# convert to gray
	img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	# get features and distances
	(kpts1, des1) = get_features(img1_gray, M, './target.jpg', show=False)
	(kpts2, des2) = get_features(img2_gray, M, './reference.jpg', show=False)
	# distances = similarity_descriptors(des1, des2)
	Hgt = Hypergraph(kpts1, des1)
	Hgr = Hypergraph(kpts2, des2)
	# Hs = get_Hs(Hgt, Hgr, kpts1, kpts2)
	# Hgt.show_hyperedges()
