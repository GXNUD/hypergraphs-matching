from __future__ import division
import cProfile
import numpy as np
from numpy import linalg as LA

def get_angle(V1, V2):
    dot = np.vdot(V1, V2)
    angle = np.arccos( dot / (LA.norm(V1) * LA.norm(V2)) )
    return np.degrees(angle)

def main():
    A = [5, 7]
    B = [10, 5]
    C = [5, 5]
    V1 = np.subtract(A, C)
    V2 = np.subtract(B, C)
    print "Angle : {0}".format(get_angle(V1, V2))

if __name__ == '__main__':
    main()
