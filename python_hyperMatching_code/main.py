from __future__ import division
import numpy as np
import random
import cv2
from hypergraph import Hypergraph
from math import log
import match


'''
    ######## ########    ###    ######## ##     ## ########  ########  ######
    ##       ##         ## ##      ##    ##     ## ##     ## ##       ##    ##
    ##       ##        ##   ##     ##    ##     ## ##     ## ##       ##
    ######   ######   ##     ##    ##    ##     ## ########  ######    ######
    ##       ##       #########    ##    ##     ## ##   ##   ##             ##
    ##       ##       ##     ##    ##    ##     ## ##    ##  ##       ##    ##
    ##       ######## ##     ##    ##     #######  ##     ## ########  ######
'''


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
    cv2.imwrite(outname, img_to_write)
    if show:
        cv2.namedWindow('Keypoints')  # , cv2.WINDOW_NORMAL)
        cv2.imshow('Keypoints', img_to_write)
        cv2.waitKey()
    return (kp[:limit], des[:limit]) if len(kp) > limit else (kp, des)


'''
    ########  ########     ###    ##      ## #### ##    ##  ######
    ##     ## ##     ##   ## ##   ##  ##  ##  ##  ###   ## ##    ##
    ##     ## ##     ##  ##   ##  ##  ##  ##  ##  ####  ## ##
    ##     ## ########  ##     ## ##  ##  ##  ##  ## ## ## ##   ####
    ##     ## ##   ##   ######### ##  ##  ##  ##  ##  #### ##    ##
    ##     ## ##    ##  ##     ## ##  ##  ##  ##  ##   ### ##    ##
    ########  ##     ## ##     ##  ###  ###  #### ##    ##  ######
'''


def draw_edges_match(matches, kp1, kp2, E1, E2, img1, img2):
    (rows1, cols1) = img1.shape
    (rows2, cols2) = img2.shape
    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

    for mat in matches:
        out[:rows1, :cols1] = np.dstack([img1, img1, img1])
        out[:rows2, cols1:] = np.dstack([img2, img2, img2])
        __queryIdx = mat[0]
        __trainIdx = mat[1]

        (x1_1, y1_1) = kp1[E1[__queryIdx][0]].pt
        (x2_1, y2_1) = kp1[E1[__queryIdx][1]].pt
        (x3_1, y3_1) = kp1[E1[__queryIdx][2]].pt

        (x1_2, y1_2) = kp2[E2[__trainIdx][0]].pt
        (x2_2, y2_2) = kp2[E2[__trainIdx][1]].pt
        (x3_2, y3_2) = kp2[E2[__trainIdx][2]].pt

        cv2.circle(out, (int(x1_1), int(y1_1)), 1, (0, 0, 255), 3)
        cv2.circle(out, (int(x2_1), int(y2_1)), 1, (0, 0, 255), 3)
        cv2.circle(out, (int(x3_1), int(y3_1)), 1, (0, 0, 255), 3)
        cv2.circle(out, (int(x1_2) + cols1, int(y1_2)), 1, (0, 255, 0), 3)
        cv2.circle(out, (int(x2_2) + cols1, int(y2_2)), 1, (0, 255, 0), 3)
        cv2.circle(out, (int(x3_2) + cols1, int(y3_2)), 1, (0, 255, 0), 3)
        print "-->", __queryIdx, __trainIdx
        print (x1_1, y1_1)
        print (x2_1, y2_1)
        print (x3_1, y3_1)
        print
        print (x1_2, y1_2)
        print (x2_2, y2_2)
        print (x3_2, y3_2)
        print
        print "similarity        -> {}".format(mat[2])
        print "similarity ratios  : {}".format(mat[3])
        print "similarity angles  : {}".format(mat[4])
        print "similarity desc    : {} {}".format(mat[5], - 0.5 * log(mat[5]))

        # cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)),
        # (102, 255, 255), 1)
        # cv2.namedWindow('Matching', cv2.WINDOW_NORMAL)
        cv2.imshow("Matching", out)
        cv2.waitKey()
        cv2.destroyWindow("Matching")
    return out


def draw_points_match(matches, kp1, kp2, img1, img2, name='Matching'):
    (rows1, cols1) = img1.shape
    (rows2, cols2) = img2.shape
    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
    out[:rows1, :cols1] = np.dstack([img1, img1, img1])
    out[:rows2, cols1:] = np.dstack([img2, img2, img2])
    for mat in matches:
        __queryIdx = mat.queryIdx
        __trainIdx = mat.trainIdx

        (x1, y1) = kp1[__queryIdx].pt
        (x2, y2) = kp2[__trainIdx].pt
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        cv2.circle(out, (int(x1), int(y1)), 1, color, 3)
        cv2.circle(out, (int(x2) + cols1, int(y2)), 1, color, 3)
        cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), color, 1)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, out)
    # cv2.waitKey()
    # cv2.destroyWindow(name)


def draw_triangulation(kp, E, img):
    rows, cols = img.shape
    out = np.zeros((rows, cols, 3), dtype='uint8')
    out = np.dstack([img, img, img])
    print E
    for i, j, k in E:
        x1, y1 = kp[i].pt
        x2, y2 = kp[j].pt
        x3, y3 = kp[k].pt

        cv2.circle(out, (int(x1), int(y1)), 8, (0, 0, 255), 1)
        cv2.circle(out, (int(x2), int(y2)), 8, (0, 255, 0), 1)
        cv2.circle(out, (int(x3), int(y3)), 8, (255, 0, 0), 1)

        cv2.line(
            out, (int(x1), int(y1)), (int(x2), int(y2)),
            (102, 255, 255), 1
        )
        cv2.line(
            out, (int(x1), int(y1)), (int(x3), int(y3)),
            (102, 255, 255), 1
        )
        cv2.line(
            out, (int(x3), int(y3)), (int(x2), int(y2)),
            (102, 255, 255), 1
        )
    # cv2.imwrite('./triangulation_gay.png', out)
    cv2.namedWindow('triangulation', cv2.WINDOW_NORMAL)
    cv2.imshow('triangulation', out)
    cv2.waitKey()
    cv2.destroyWindow('triangulation')


'''
    ##     ##    ###    #### ##    ##
    ###   ###   ## ##    ##  ###   ##
    #### ####  ##   ##   ##  ####  ##
    ## ### ## ##     ##  ##  ## ## ##
    ##     ## #########  ##  ##  ####
    ##     ## ##     ##  ##  ##   ###
    ##     ## ##     ## #### ##    ##
'''


if __name__ == "__main__":
    M = 50
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

    # Show results
    # draw_edges_match(
    #    edge_matches, kpts1, kpts2, Hgt.E, Hgr.E, img1, img2
    # )
    draw_points_match(point_matches, kpts1, kpts2, img1, img2)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    draw_points_match(
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
