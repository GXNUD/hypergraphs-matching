from __future__ import division
import cv2
import numpy as np
from numpy import linalg as LA
from math import exp
from similarity import similarity
SIGMA = 0.5


def match(E1, E2, kpts1, kpts2, desc1, desc2, cang, crat, cdesc, th_e, th_p):
    '''
    E1, E2: hyperedges lists of img1 and img2, respectly
    '''
    # indices_taken = []
    hyperedge_matches = []
    point_matches = []
    sel_point_matches = set()

    for i, e_i in enumerate(E1):
        max_similarity = -float('inf')
        for j, e_j in enumerate(E2):
            p = [np.array(kpts1[e_i[k]].pt) for k in xrange(3)]
            q = [np.array(kpts2[e_j[k]].pt) for k in xrange(3)]
            dp = [np.array(desc1[e_i[k]]) for k in xrange(3)]
            dq = [np.array(desc2[e_j[k]]) for k in xrange(3)]
            _point_idx, _sim, sim_a, sim_r, sim_d = similarity(
                p, q,
                dp, dq,
                cang, crat, cdesc
            )
            if _sim > max_similarity:
                best_index = j
                max_similarity = _sim
                s_ang = sim_a
                s_ratios = sim_r
                s_desc = sim_d
                e_idx = [(e_i[i], e_j[j]) for i, j in _point_idx]
        if max_similarity >= th_e:
            hyperedge_matches.append(
                (i, best_index, max_similarity, s_ang, s_ratios, s_desc)
            )
            for i, j in e_idx:
                dist = LA.norm(np.subtract(desc1[i], desc2[j]))
                sim = exp(-dist / SIGMA)
                if not (i, j) in sel_point_matches and sim >= th_p:
                    point_matches.append(cv2.DMatch(i, j, dist))
                    sel_point_matches.add((i, j))
    return hyperedge_matches, point_matches
