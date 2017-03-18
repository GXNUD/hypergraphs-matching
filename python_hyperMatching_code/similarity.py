from __future__ import division
from math import exp, sin
from numpy import linalg as LA
import numpy as np
SIGMA = 0.5
PERMS = [
    [(0, 0), (1, 1), (2, 2)],
    [(0, 0), (1, 2), (2, 1)],
    [(0, 1), (1, 0), (2, 2)],
    [(0, 1), (1, 2), (2, 0)],
    [(0, 2), (1, 0), (2, 1)],
    [(0, 2), (1, 1), (2, 0)]
]


def angle(p, i):
    a = np.subtract(p[i], p[(i + 1) % 3])
    b = np.subtract(p[i], p[(i + 2) % 3])
    return np.arccos(np.dot(a, b) / LA.norm(a) / LA.norm(b))


def sim_angles(p, q, idx1, idx2, idx3):
    i1, j1 = idx1
    i2, j2 = idx2
    i3, j3 = idx3
    return exp(-np.mean([
        abs(sin(angle(p, i1)) - sin(angle(q, j1))),
        abs(sin(angle(p, i2)) - sin(angle(q, j2))),
        abs(sin(angle(p, i3)) - sin(angle(q, j3)))
    ]) / SIGMA)


def opposite_side(p, i):
    return LA.norm(np.subtract(p[(i + 1) % 3], p[(i + 2) % 3]))


def sim_ratios(p, q, idx1, idx2, idx3):
    i1, j1 = idx1
    i2, j2 = idx2
    i3, j3 = idx3
    R1 = opposite_side(p, i1) / opposite_side(q, j1)
    R2 = opposite_side(p, i2) / opposite_side(q, j2)
    R3 = opposite_side(p, i3) / opposite_side(q, j3)
    return exp(-np.std([R1, R2, R3]) / SIGMA)


def sim_desc(dp, dq, idx1, idx2, idx3):
    i1, j1 = idx1
    i2, j2 = idx2
    i3, j3 = idx3
    return exp(-np.mean([
        LA.norm(np.subtract(dp[i1], dq[j1])),
        LA.norm(np.subtract(dp[i2], dq[j2])),
        LA.norm(np.subtract(dp[i3], dq[j3]))
    ]) / SIGMA)


def similarity(p, q, dp, dq, cang, crat, cdesc):
    s = cang + crat + cdesc
    cang /= s
    crat /= s
    cdesc /= s

    max_sim = -float('inf')
    for idx in PERMS:
        _sim_a = sim_angles(p, q, *idx)
        _sim_r = sim_ratios(p, q, *idx)
        _sim_d = sim_desc(dp, dq, *idx)
        sim = cang * _sim_a + crat * _sim_r + cdesc * _sim_d
        if sim > max_sim:
            max_sim = sim
            point_match = idx
            sim_a = _sim_a
            sim_r = _sim_r
            sim_d = _sim_d
    return point_match, max_sim, sim_a, sim_r, sim_d
