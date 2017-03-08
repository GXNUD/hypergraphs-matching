from __future__ import division
import cv2
from math import exp, log
import numpy as np
import similarity as sim


def hyperedges(E1, E2, kpts1, kpts2, desc1, desc2, cang, crat, cdesc, th):
    '''
    E1, E2: hyperedges lists of img1 and img2, respectly
    '''
    sigma = 0.5
    # indices_taken = []
    matches = []

    s = cang + crat + cdesc
    cang /= s
    crat /= s
    cdesc /= s

    for i, e_i in enumerate(E1):
        best_index = -float('inf')
        max_similarity = -float('inf')
        s_ang = -float('inf')
        s_ratios = -float('inf')
        s_desc = -float('inf')
        for j, e_j in enumerate(E2):
            p = [np.array(kpts1[e_i[k]].pt) for k in xrange(3)]
            q = [np.array(kpts2[e_j[k]].pt) for k in xrange(3)]
            dp = [np.array(desc1[e_i[k]]) for k in xrange(3)]
            dq = [np.array(desc2[e_j[k]]) for k in xrange(3)]

            sim_ratios = sim.ratios(p, q, sigma)
            sim_angles = sim.angles(p, q, sigma)
            sim_desc = sim.descriptors(dp, dq, sigma)
            _sim = cang * sim_angles + crat * sim_ratios + cdesc * sim_desc

            if _sim > max_similarity:
                best_index = j
                max_similarity = _sim
                s_ratios = sim_ratios
                s_ang = sim_angles
                s_desc = sim_desc
        if max_similarity >= th:
            matches.append(
                (i, best_index, max_similarity, s_ratios, s_ang, s_desc)
            )
        # indices_taken.append(best_index)
    return matches


def points(matches, desc1, desc2, E1, E2, th, sigma=0.5):
    matched_points = []
    S = set()
    for mat in matches:
        __queryIdx = mat[0]
        __trainIdx = mat[1]

        p_desc = [np.array(desc1[E1[__queryIdx][i]]) for i in xrange(3)]
        q_desc = [np.array(desc2[E2[__trainIdx][i]]) for i in xrange(3)]

        sim_desc = [
            [exp(-sum(np.abs(q - p)) / sigma) for q in q_desc] for p in p_desc
        ]
        best_match = [s.index(max(s)) for s in sim_desc]

        for i, j in enumerate(best_match):
            qI, tT = E1[__queryIdx][i], E2[__trainIdx][j]
            if not (qI, tT) in S and sim_desc[i][j] >= th:
                matched_points.append(
                    cv2.DMatch(qI, tT, -sigma * log(sim_desc[i][j]))
                )
                S.add((qI, tT))
    return matched_points
