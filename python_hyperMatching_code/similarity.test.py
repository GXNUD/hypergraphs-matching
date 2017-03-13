from __future__ import division
import math
import matplotlib.pyplot as plt
from similarity import similarity
import numpy as np


def rot(p, t):
    x, y = p
    return (
        x * math.cos(t) - y * math.sin(t), x * math.sin(t) + y * math.cos(t)
    )


def trans(p, tx, ty):
    x, y = p
    return (x + tx, y + ty)


if __name__ == '__main__':
    p = [
        np.array([119.91277313232422, 252.8047332763672]),
        np.array([139.75482177734375, 284.2823181152344]),
        np.array([109.2437744140625, 257.5870056152344])
    ]
    dp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # q = [trans(rot(pp, 10 * math.pi / 180), 2, 3) for pp in p]
    q = [
        np.array([374.65393066, 144.14517212]),
        np.array([161.24133301, 217.84576416]),
        np.array([99.27768707, 161.40586853])
    ]
    dq = [[1, 2, 3.1], [4, 5.1, 6], [7.1, 8, 9]]

    point_match, max_sim, sim_a, sim_r, sim_d = similarity(
        p, q,
        dp, dq,
        1, 1, 0.4, True
    )
    print p
    print q
    print 'max sim    ->  {}'.format(max_sim)
    print 'sim angles  :  {}'.format(sim_a)
    print 'sim ratios  :  {}'.format(sim_r)
    print 'sim desc    :  {}'.format(sim_d)

    xp, yp = zip(*p)
    xq, yq = zip(*q)
    plt.plot(xp + (xp[0],), yp + (yp[0],), 'r-')
    plt.plot(xq + (xq[0],), yq + (yq[0],), 'b-')
    plt.xlim((min(xp + xq), max(xp + xq)))
    plt.ylim((min(yp + yq), max(yp + yq)))
    plt.gca().invert_yaxis()
    plt.show()
