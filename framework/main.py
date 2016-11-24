from math import sqrt, pow, exp
import numpy as np
from numpy import linalg as LA
import cv2

#==============================================================================#

class Hypergraph():
	def __init__(self, kpts, desc):
		self.kpts = kpts
		self.desc = desc
		self.size = len(kpts)
		self.V = [i for i in xrange(len(kpts))]
		self.E = self.__getHyperedges(kpts)

	def __getHyperedges(self, kpts):
		'''
		builds the hyperedges using the 2 nearest neighbors of each point
		'''
		E = []
		n = len(kpts)
		dist = np.zeros((n, n))
		W = compute_w_parameter(self.kpts)
		# print "W: ", W
		for i, r in enumerate(kpts):
			for j, l in enumerate(kpts):
				dist[i][j] = self.__euclideanDistance(r, l)
		for i, values in enumerate(dist):
			min1, min2  = self.__getMinValues(values, i)
			edge_weigth = hyperedge_weigth((i, min1, min2), 1, self.kpts)
			E.append((i, min1, min2, edge_weigth))
		return E

	def __euclideanDistance(self, kpt1, kpt2):
		return sqrt(pow(kpt2.pt[0] - kpt1.pt[0], 2) + pow(kpt2.pt[1] - kpt1.pt[1], 2))

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

#==============================================================================#

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
	'''
	A measurement of similarity between two Keypoint's descriptors
	'''
	norm1 = LA.norm(des1)
	norm2 = LA.norm(des2)
	similarity =  np.dot(des1, des2) / (norm1 * norm2)
	return similarity

def similarity_descriptors(target_descriptors, ref_descriptors):
	n = len(target_descriptors)
	m = len(ref_descriptors)
	dist = np.zeros((n, m))
	for i, t in enumerate(target_descriptors):
		for j, r in enumerate(ref_descriptors):
			dist[i][j] = similarity(t, r)
			# print "similarity between {} and {}".format(i, j), dist[i][j]
	return dist

#==============================================================================#

def hyperedge_weigth(edge, W, kpts):
	'''
	det([vi - vk, vj - vk]) * sum 1 / sqrt(norm((vi - vk) * norm(vj - vk)))
	'''
	vi = np.asarray(kpts[edge[0]].pt)
	vj = np.asarray(kpts[edge[1]].pt)
	vk = np.asarray(kpts[edge[2]].pt)
	mat = np.array([vi - vk, vj - vk])
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
				if vi.all() != vk.all() and vj.all() != vk.all():
					acum += 1 / sqrt(LA.norm(vi - vk) * LA.norm(vj - vk)) # TODO FIX division
	return acum

# def hyperedges_similarity(ei, ej, w1, w2, kpts1, kpts2):
# 	a = hyperedge_weigth(ei, w1, kpts1)
# 	b = hyperedge_weigth(ej, w2, kpts2)
# 	sigma = 0.5 # TODO wich is the best sigma
# 	S = exp(-( pow(LA.norm(a - b), 2) / sigma ))
# 	return S

#=========================== Angle diff Utils =================================#

def vectors_angle(pivot, p, q):
	V1 = np.subtract(p, pivot)
	V2 = np.subtract(q, pivot)
	dot = np.vdot(V1, V2)
	angle = np.arccos( dot / (LA.norm(V1) * LA.norm(V2)) )
	return np.degrees(angle)

def get_angles(edge, kpts): #TODO fix the nan values
	print "\nEDGE: {}".format(edge)
	p1, p2, p3 = kpts[edge[0]].pt, kpts[edge[1]].pt, kpts[edge[2]].pt
	print "Coordenates p1: {}".format(p1)
	print "Coordenates p2: {}".format(p2)
	print "Coordenates p3: {}".format(p3)
	alpha = vectors_angle(p1, p2, p3) # angle of vectors p2 - p1, p3 - p1
	beta  = vectors_angle(p2, p1, p3) # angle of vectors p1 - p2, p3 - p2
	theta = vectors_angle(p3, p1, p2) # angle of vectors p1 - p3, p2 - p3
	return [alpha, beta, theta]

def similarity_angles(e1, e2, kpts1, kpts2, sigma=0.01):
	'''

	angles1 and angles2 are those formed by the first and second triangle, respecvitly
	'''
	print "\n\n#==================================================================#"
	sum_diff_between_angles = 0
	angles1 = get_angles(e1, kpts1)
	angles2 = get_angles(e2, kpts2)
	print "Angles 1: {}".format(angles1)
	print "Angles 2: {}".format(angles2)
	for i in xrange(3):
		sum_diff_between_angles += (angles1[i] - angles2[i])
	similarity = exp(- sum_diff_between_angles / sigma)
	print "similarity: {}".format(similarity)
	return similarity

#=========================== Angle diff Utils =================================#

def match_hyperedges(E1, E2, kpts1, kpts2):
	'''
	E1, E2: hyperedges lists of img1 and img2, respectly
	'''
	sigma = 0.01
	indices_taken = []
	matches = []
	for i, e_i in enumerate(E1):
		best_index_taken = -1
		max_similarity   = -1
		for j, e_j in enumerate(E2):
			# similarity = exp(-( pow(LA.norm(e_i[3] - e_j[3]), 2) / sigma ))
			similarity = similarity_angles(e_j, e_i, kpts1, kpts2, sigma)
			if similarity > max_similarity and not j in indices_taken:
				max_similarity = similarity
				best_index = j
		matches.append((i, best_index))
		indices_taken.append(best_index)
	return matches

# def refine_matches(edge_matches, kpts1, kpts2, E1, E2, dists):
# 	'''
# 	edge_matches: first match by hyperedge similarity
# 	dists: distances between both images points
# 	'''
# 	for m_id, match in enumerate(edge_matches):
# 		for i, node_i in enumerate(E1[match[0]]):
# 			best
# 			for j, node_j in enumerate(E2[match[1]]):

def draw_edges_match(matches, kp1, kp2, E1, E2, img1, img2):
	(rows1, cols1) = img1.shape
	(rows2, cols2) = img2.shape
	out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
	for mat in matches:
		out[:rows1, :cols1] = np.dstack([img1, img1, img1])
		out[:rows2, cols1:] = np.dstack([img2, img2, img2])
		__queryIdx = mat[0]
		__trainIdx = mat[1]

		(x1_1, y1_1) = kp1[E1[__queryIdx][0]].pt
		(x2_1, y2_1) = kp1[E1[__queryIdx][1]].pt
		(x3_1, y3_1) = kp1[E1[__queryIdx][2]].pt
		(x1_2, y1_2) = kp1[E1[__trainIdx][0]].pt
		(x2_2, y2_2) = kp1[E1[__trainIdx][1]].pt
		(x3_2, y3_2) = kp1[E1[__trainIdx][2]].pt
		cv2.circle(out, (int(x1_1), int(y1_1)), 8, (0, 0, 255), 1)
		cv2.circle(out, (int(x2_1), int(y2_1)), 8, (0, 0, 255), 1)
		cv2.circle(out, (int(x3_1), int(y3_1)), 8, (0, 0, 255), 1)
		cv2.circle(out, (int(x1_2) + cols1, int(y1_2)), 8, (0, 255, 0), 1)
		cv2.circle(out, (int(x2_2) + cols1, int(y2_2)), 8, (0, 255, 0), 1)
		cv2.circle(out, (int(x3_2) + cols1, int(y3_2)), 8, (0, 255, 0), 1)
		# cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (102, 255, 255), 1)
		# cv2.namedWindow('Matching', cv2.WINDOW_NORMAL)
		cv2.imshow("Matching", out)
		cv2.waitKey()
		cv2.destroyWindow("Matching")
	return out

if __name__ == "__main__":
	M = 15
	img1 = cv2.imread('./house.seq0.png')
	img2 = cv2.imread('./house.seq27.png')
	# convert to gray
	img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	# get features and distances between every pair of points from both images
	(kpts1, des1) = get_features(img1_gray, M, './target.jpg', show=False)
	(kpts2, des2) = get_features(img2_gray, M, './reference.jpg', show=False)
	distances = similarity_descriptors(des1, des2)

	Hgt = Hypergraph(kpts1, des1)
	Hgr = Hypergraph(kpts2, des2)

	# match points
	matches = match_hyperedges(Hgt.E, Hgr.E, kpts1, kpts2)

	# show results
	# draw_edges_match(matches, kpts1, kpts2, Hgt.E, Hgr.E, img1_gray, img2_gray)
