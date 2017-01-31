from __future__ import division
import numpy as np
import random
import cv2
from hypergraph import Hypergraph
from math import exp
import similarity as sim


'''
    ######## ########    ###    ######## ##     ## ########  ########  ######
    ##       ##         ## ##      ##    ##     ## ##     ## ##       ##    ##
    ##       ##        ##   ##     ##    ##     ## ##     ## ##       ##
    ######   ######   ##     ##    ##    ##     ## ########  ######    ######
    ##       ##       #########    ##    ##     ## ##   ##   ##             ##
    ##       ##       ##     ##    ##    ##     ## ##    ##  ##       ##    ##
    ##       ######## ##     ##    ##     #######  ##     ## ########  ######
'''


def get_features(img, limit=None, outname="sample.jpg", show=False):
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
    ##     ##    ###    ########  ######  ##     ## #### ##    ##  ######
    ###   ###   ## ##      ##    ##    ## ##     ##  ##  ###   ## ##    ##
    #### ####  ##   ##     ##    ##       ##     ##  ##  ####  ## ##
    ## ### ## ##     ##    ##    ##       #########  ##  ## ## ## ##   ####
    ##     ## #########    ##    ##       ##     ##  ##  ##  #### ##    ##
    ##     ## ##     ##    ##    ##    ## ##     ##  ##  ##   ### ##    ##
    ##     ## ##     ##    ##     ######  ##     ## #### ##    ##  ######
'''


def match_hyperedges(E1, E2, kpts1, kpts2, desc1, desc2, c1, c2, c3, th):
    '''
    E1, E2: hyperedges lists of img1 and img2, respectly
    '''
    sigma = 0.5
    # indices_taken = []
    matches = []

    s = sum([c1, c2, c3])
    c1 /= s
    c2 /= s
    c3 /= s

    for i, e_i in enumerate(E1):
        best_index = -float('inf')
        max_similarity = -float('inf')
        s_ang = -float('inf')
        s_area = -float('inf')
        s_desc = -float('inf')
        for j, e_j in enumerate(E2):
            p = [np.array(kpts1[e_i[k]].pt) for k in xrange(3)]
            q = [np.array(kpts2[e_j[k]].pt) for k in xrange(3)]
            dp = [np.array(desc1[e_i[k]]) for k in xrange(3)]
            dq = [np.array(desc2[e_j[k]]) for k in xrange(3)]

            sum_area = sim.ratios(p, q, sigma)
            sim_angles = sim.angles(p, q, sigma)
            sim_desc = sim.descriptors(dp, dq, sigma)
            similarity = c1 * sum_area + c2 * sim_angles + c3 * sim_desc
            # exit()
            if similarity > max_similarity:
                best_index = j
                max_similarity = similarity
                s_area = sum_area
                s_ang = sim_angles
                s_desc = sim_desc
        if max_similarity >= th:
            matches.append(
                (i, best_index, max_similarity, s_area, s_ang, s_desc)
            )
        # indices_taken.append(best_index)
    return matches


def match_points(matches, desc1, desc2, E1, E2, th, sigma=0.5):
    matched_points = []
    for mat in matches:
        __queryIdx = mat[0]
        __trainIdx = mat[1]

        p_desc = [np.array(desc1[E1[__queryIdx][i]]) for i in xrange(3)]
        q_desc = [np.array(desc2[E2[__trainIdx][i]]) for i in xrange(3)]

        sim_desc = [
            [exp(-sum(np.abs(q - p)) / sigma) for q in q_desc] for p in p_desc
        ]
        best_match = [s.index(max(s)) for s in sim_desc]

        for i, j in enumerate(best_match):
            if sim_desc[i][j] >= th:
                matched_points.append(
                    cv2.DMatch(
                        E1[__queryIdx][i], E2[__trainIdx][j], sim_desc[i][j]
                    )
                )

    return matched_points


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
        print "similarity desc    : {}".format(mat[5])

        # cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)),
        # (102, 255, 255), 1)
        # cv2.namedWindow('Matching', cv2.WINDOW_NORMAL)
        cv2.imshow("Matching", out)
        cv2.waitKey()
        cv2.destroyWindow("Matching")
    return out


def draw_points_match(matches, kp1, kp2, img1, img2):
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
    cv2.namedWindow('Matching', cv2.WINDOW_NORMAL)
    cv2.imshow("Matching", out)
    cv2.waitKey()
    cv2.destroyWindow("Matching")


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
    img1 = cv2.imread('./img/house.seq80.png')
    img2 = cv2.imread('./img/house.seq80.180.png')
    # convert to gray
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # get features and distances between every pair of points from both images
    (kpts1, des1) = get_features(img1_gray, M, './target.jpg', show=False)
    (kpts2, des2) = get_features(img2_gray, M, './reference.jpg', show=False)

    # distances = similarity_descriptors(des1, des2)
    Hgt = Hypergraph(kpts1, des1)
    Hgr = Hypergraph(kpts2, des2)
    print 'Hypergraph construction done'
    # matching
    edge_matches = match_hyperedges(
        Hgt.E, Hgr.E, kpts1, kpts2, des1, des2, 10, 7, 5, 0.6
    )
    print 'Hyperedges matching done'

    point_matches = match_points(edge_matches, des1, des2, Hgt.E, Hgr.E, 0.7)
    # show results
    draw_edges_match(
       edge_matches, kpts1, kpts2, Hgt.E, Hgr.E, img1_gray, img2_gray
    )
    draw_points_match(point_matches, kpts1, kpts2, img1_gray, img2_gray)
