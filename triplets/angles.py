from __future__ import division
import numpy as np
from numpy import linalg as LA

def getAngle1(V1, V2):
    '''
    This works
    arccos( (V1 dot V2) / ( norm(V1) * norm(v2) ))
    '''
    dot = np.vdot(V1, V2)
    angle = np.arccos( dot / (LA.norm(V1) * LA.norm(V2)) )
    return np.degrees(angle)

def getAngle2(V1, V2):
    '''
    arcsin( (V1 cross V2) / ( norm(V1) * norm(v2) ) )
    '''
    cross = np.cross(V1, V2)
    angle = np.arcsin( cross / (LA.norm(V1) * LA.norm(V2)) )
    return np.degrees(angle)

def main():
    # A = [2, 5]
    # B = [7, 2]
    # C = [2, 2]
    A = [5, 7]
    B = [10, 5]
    C = [5, 5]
    V1 = np.subtract(A, C)
    V2 = np.subtract(B, C)
    print "With arcsin", getAngle2(V1, V2)
    print "With arccos", getAngle1(V1, V2)

if __name__ == '__main__':
    main()
