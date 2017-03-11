from __future__ import division
import math
import matplotlib.pyplot as plt
from similarity import similarity


def rot(p, t):
    x, y = p
    return (
        x * math.cos(t) - y * math.sin(t), x * math.sin(t) + y * math.cos(t)
    )


def trans(p, tx, ty):
    x, y = p
    return (x + tx, y + ty)


if __name__ == '__main__':
    p = [(0, 0), (1, 1), (-1, 1)]
    dp = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    q = [trans(rot(pp, 10 * math.pi / 180), 2, 3) for pp in p]
    # q = [(1, 0), (1, -1), (3, 2)]
    dq = [[4, 8, 3], [7, 5, 0], [1, 4, 6]]

    print p
    print q
    point_match, max_sim, sim_a, sim_r, sim_d = similarity(
        p, q,
        dp, dq,
        1, 1, 2
    )
    print 'max sim:     {}'.format(max_sim)
    print 'sim angles:  {}'.format(sim_a)
    print 'sim ratios:  {}'.format(sim_r)
    print 'sim desc:    {}'.format(sim_d)

    xp, yp = zip(*p)
    xq, yq = zip(*q)
    plt.gca().invert_yaxis()
    plt.plot(xp + (xp[0],), yp + (yp[0],), 'r-')
    plt.plot(xq + (xq[0],), yq + (yq[0],), 'b-')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.show()
