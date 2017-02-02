from __future__ import division
from math import exp, sin
from numpy import linalg as LA
import numpy as np
import itertools


def vectors_angle_sin(pivot, p, q):
    V1 = np.subtract(p, pivot)
    V2 = np.subtract(q, pivot)
    dot = np.dot(V1, V2)
    angle = np.arccos(dot / (LA.norm(V1) * LA.norm(V2)))
    return sin(angle)


def get_angles_sin(p):
    p1, p2, p3 = p
    alpha = vectors_angle_sin(p1, p2, p3)
    beta = vectors_angle_sin(p2, p1, p3)
    theta = vectors_angle_sin(p3, p1, p2)

    return [alpha, beta, theta]


# def angles(e1, e2, kpts1, kpts2, sigma=0.5):
def angles(p, q, sigma=0.5):
    '''
    angles1 and angles2 are those formed by the first and second triangle,
    respecvitly, we'll choose the best case scenario for the sum of sin
    '''

    # sin1 = get_angles_sin(e1, kpts1)
    # sin2 = get_angles_sin(e2, kpts2)

    sin1 = get_angles_sin(p)
    sin2 = get_angles_sin(q)

    perms = itertools.permutations(sin1)
    diffs = [sum(np.abs(np.subtract(s, sin2))) for s in perms]
    min_diff_between_sin = min(diffs)
    similarity = exp(-min_diff_between_sin / sigma)

    return similarity


def ratios(p, q, sigma=0.5):
    dp = [
        LA.norm(np.subtract(p[i], p[j]))
        for i, j in itertools.combinations(xrange(3), 2)
    ]
    dq = [
        LA.norm(np.subtract(q[i], q[j]))
        for i, j in itertools.combinations(xrange(3), 2)
    ]

    perms = itertools.permutations(dq)

    err = float('inf')
    for sides_q in perms:
        sides_p = dp
        r = [a / A for a, A in zip(sides_p, sides_q)]
        diffs = [
            np.abs(r[i] - r[j])
            for i, j in itertools.combinations(xrange(3), 2)
        ]
        _err = sum(diffs)
        err = min(err, _err)

    similarity = exp(-err / sigma)
    return similarity


def descriptors(p, q, sigma=0.5):
    perms = itertools.permutations(q)

    diffs = [
        # sum([LA.norm(np.subtract(qqi, pi)) for qqi, pi in zip(qq, p)])
        # for qq in perms
        sum([LA.norm(np.subtract(qqi, pi)) for qqi, pi in zip(qq, p)])
        for qq in perms
    ]

    min_diff = min(diffs)
    similarity = exp(- min_diff / sigma)

    return similarity
