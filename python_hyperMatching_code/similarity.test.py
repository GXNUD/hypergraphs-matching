from __future__ import division
import math
import matplotlib.pyplot as plt
import similarity as sim


def rot(p, t):
    x, y = p
    return (
        x * math.cos(t) - y * math.sin(t), x * math.sin(t) + y * math.cos(t)
    )


def trans(p, tx, ty):
    x, y = p
    return (x + tx, y + ty)


if __name__ == '__main__':
    p = [(1, 0), (3, 4), (5, 2)]
    q = [trans(rot(pp, 56.43 * math.pi / 180), 2, -1) for pp in p]

    print p
    print q
    print sim.angles(p, q)
    print sim.ratios(p, q)
    print sim.descriptors([1, 2, 3, 4], [2, 4, 1, 3])

    xp, yp = zip(*p)
    xq, yq = zip(*q)
    plt.plot(xp + (xp[0],), yp + (yp[0],), 'r-')
    plt.plot(xq + (xq[0],), yq + (yq[0],), 'b-')
    plt.xlim((-3, 6))
    plt.ylim((-3, 6))
    plt.show()
