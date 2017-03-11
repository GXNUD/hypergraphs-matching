#!/usr/bin/env python
from __future__ import division
import numpy as np
import cv2
from hypergraph import Hypergraph
from match import match
import draw
import sys
import getopt


def get_features(img, limit=None, outname='sample.jpg', show=False):
    '''
        img should be gray
    '''
    detector = cv2.FeatureDetector_create('SURF')
    descriptor = cv2.DescriptorExtractor_create('SURF')
    kp = detector.detect(img)
    # getting most relevant points
    kp = sorted(kp, key=lambda x: x.response, reverse=True)
    kp, des = descriptor.compute(img, kp)
    img_to_write = np.zeros(img.shape)
    img_to_write = cv2.drawKeypoints(img, kp[:limit], img_to_write)
    if show:
        cv2.imwrite(outname, img_to_write)
        cv2.namedWindow('Keypoints')  # , cv2.WINDOW_NORMAL)
        cv2.imshow('Keypoints', img_to_write)
        cv2.waitKey()
    return (kp[:limit], des[:limit]) if len(kp) > limit else (kp, des)


def do_match(img1, img2, cang, crat, cdesc):
    M = None

    # Get features and distances between every pair of points from both images
    (kpts1, des1) = get_features(img1, M, 'target.jpg')
    (kpts2, des2) = get_features(img2, M, 'reference.jpg')

    Hgt = Hypergraph(kpts1, des1)
    Hgr = Hypergraph(kpts2, des2)

    # draw.triangulation(kpts1, Hgt.E, img1, 'Triangulation 1')
    # draw.triangulation(kpts2, Hgr.E, img2, 'Triangulation 2')

    print 'Hypergraph construction done'
    edge_matches, point_matches = match(
        Hgt.E, Hgr.E, kpts1, kpts2, des1, des2,
        cang, crat, cdesc,
        0.7, 0.75, True
    )
    print 'Hyperedges matching done'

    # draw.edges_match(edge_matches, kpts1, kpts2, Hgt.E, Hgr.E, img1, img2)

    point_matches = sorted(point_matches, key=lambda x: x.distance)
    draw.points_match(point_matches, kpts1, kpts2, img1, img2)

    cv2.waitKey()
    cv2.destroyAllWindows()


def cright():
    '''
        Prints copyright and some info
    '''
    print 'Sample implementation of LYSH algorithm for image matching'
    print 'Copyright (C) 2016 L.A. Campeon, Y.H. Gomez, J.S. Vega.'
    print 'This is free software; see the source code for copying conditions.'
    print 'There is ABSOLUTELY NO WARRANTY; not even for MERCHANTABILITY or'
    print 'FITNESS FOR A PARTICULAR PURPOSE.'
    print


def usage():
    '''
        Prints the correct usage for the program
    '''
    opts = ['--cang', '--crat', '--cdesc']
    description = [
        'Constant of angle similarity (default: 1)',
        'Constant of ratio similarity (default: 1)',
        'Constant of SURF descriptor similarity (default: 1)'
    ]

    print 'Usage: {} [options ...] img1 img2'.format(sys.argv[0])
    print
    print 'Matching options:'
    for o, d in zip(opts, description):
        print '  {}: {}'.format(o, d)
    sys.exit(2)


def main(argv):
    short_opts = 'h'
    long_opts = ['help', 'cang=', 'crat=', 'cdesc=']

    try:
        opts, args = getopt.getopt(argv, short_opts, long_opts)
    except getopt.GetoptError:
        usage()

    cang = 1.0
    crat = 1.0
    cdesc = 1.0

    try:
        for opt, arg in opts:
            if opt == '-h' or opt == '--help':
                cright()
                usage()
            elif opt == '--cang':
                cang = float(arg)
            elif opt == '--crat':
                crat = float(arg)
            elif opt == '--cdesc':
                cdesc = float(arg)
    except ValueError:
        usage()

    if len(args) != 2:
        print 'Error: You must provide two images'
        print
        usage()

    img1, img2 = args
    img1 = cv2.imread(img1, 0)
    img2 = cv2.imread(img2, 0)

    if (img1 is None) or (img1 is None):
        print 'Error: img1 and img2 must be valid images both'
        print
        usage()

    do_match(img1, img2, cang, crat, cdesc)


if __name__ == "__main__":
    main(sys.argv[1:])
