from __future__ import division
import numpy as np
import cv2
from hypergraph import Hypergraph
import match
import draw
import sys


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


def main(argv):
    M = 20
    img1 = cv2.imread('./img/house.seq80.png', 0)
    img2 = cv2.imread('./img/house.seq80.rot.png', 0)

    # Get features and distances between every pair of points from both images
    (kpts1, des1) = get_features(img1, M, 'target.jpg')
    (kpts2, des2) = get_features(img2, M, 'reference.jpg')

    Hgt = Hypergraph(kpts1, des1)
    Hgr = Hypergraph(kpts2, des2)

    print 'Hypergraph construction done'

    # Matching
    edge_matches = match.hyperedges(
        Hgt.E, Hgr.E, kpts1, kpts2, des1, des2, 1, 1, 2, 0.4
    )

    print 'Hyperedges matching done'

    point_matches = match.points(edge_matches, des1, des2, Hgt.E, Hgr.E, 0.1)
    point_matches = sorted(point_matches, key=lambda x: x.distance)

    draw.points_match(point_matches, kpts1, kpts2, img1, img2)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    draw.points_match(
        matches[:len(point_matches)], kpts1, kpts2, img1, img2, 'cv2'
    )

    cv2.waitKey()
    cv2.destroyAllWindows()

    print len(point_matches)
    for p in point_matches:
        print p.queryIdx, p.trainIdx, p.distance

    print len(matches)
    for p in matches:
        print p.queryIdx, p.trainIdx, p.distance


if __name__ == "__main__":
    main(sys.argv[1:])
