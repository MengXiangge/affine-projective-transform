# Author: TKF
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_proj_xform(matches,features1,features2,image1,image2):
    """
    Computer Vision 600.461/661 Assignment 2
    Args:
        matches (list of tuples): list of index pairs of possible matches. For example, if the 4-th feature in feature_coords1 and the 0-th feature
                                  in feature_coords2 are determined to be matches, the list should contain (4,0).
        features1 (list of tuples) : list of feature coordinates corresponding to image1
        features2 (list of tuples) : list of feature coordinates corresponding to image2
        image1 (numpy.ndarray): The input image corresponding to features_coords1
        image2 (numpy.ndarray): The input image corresponding to features_coords2
    Returns:
        proj_xform (numpy.ndarray): a 3x3 Projective transformation matrix between the two images, computed using the matches.
    """
    a1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    a2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    shape = (8, 9)
    a = np.zeros(shape, np.float64)
    h = np.zeros((20000, 3, 3), np.float64)
    q = np.zeros((20000, 1), np.float64)
    inlier = list()
    if len(matches)<5:
        print 'Too few matchs to calculate projective transform'
        proj_xform = np.hstack((a1, a2))
        for i in xrange(len(matches)):
            po1 = features1[matches[i][0]]
            po2 = features2[matches[i][1]]
            cv2.line(proj_xform, (po1[1], po1[0]), (po2[1] + a1.shape[1], po2[0]), (255, 0, 0), 1)
            plt.plot(po1[1], po1[0], 'r^')
            plt.plot(po2[1] + a1.shape[1], po2[0], 'r^')
        plt.imshow(proj_xform)
        plt.show()
        return np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    for i in xrange(20000):
        inlier.append([])
        index = np.random.random_integers(0, len(matches) - 1, 4)
        while (len(set(index)) < len(index)):
            index = np.random.random_integers(0, len(matches) - 1, 4)
        for j in xrange(4):
            po1 = features1[matches[index[j]][0]]
            po2 = features2[matches[index[j]][1]]
            a[2 * j] = [po1[1], po1[0], 1, 0, 0, 0, -po2[1] * po1[1], -po2[1] * po1[0], -po2[1]]
            a[2 * j + 1] = [0, 0, 0, po1[1], po1[0], 1, -po2[0] * po1[1], -po2[0] * po1[0], -po2[0]]
        _, _, ei = np.linalg.svd(a)
        min_vec = ei[-1]
        min_vec = min_vec / min_vec[-1]
        r = np.reshape(min_vec, (3, 3))
        h[i] = r
        for k in xrange(len(matches)):
            po1 = features1[matches[k][0]]
            po2 = features2[matches[k][1]]
            x1 = np.array([po1[1], po1[0], 1])
            x2 = np.dot(x1, r)
            if np.sqrt((x2[0] - po2[1]) ** 2 + (x2[1] - po2[0]) ** 2) < 20:
                q[i] += 1
                inlier[i].append(k)
    i = np.where(q == max(q))
    inl = inlier[i[0][0]]
    result=h[i[0][0]]
    proj_xform = np.hstack((image1, image2))
    for i in xrange(len(matches)):
        po1 = features1[matches[i][0]]
        po2 = features2[matches[i][1]]
        if i in inl:
            cv2.line(proj_xform, (po1[1], po1[0]), (po2[1] + image1.shape[1], po2[0]), (0, 255, 0), 1)
            cv2.circle(proj_xform,(po1[1], po1[0]),3,(0,255,0),3)
            cv2.circle(proj_xform, (po2[1]+image1.shape[1], po2[0]), 3, (0, 255, 0), 3)
        else:
            cv2.line(proj_xform, (po1[1], po1[0]), (po2[1] + image1.shape[1], po2[0]), (255, 0, 0), 1)
            cv2.circle(proj_xform, (po1[1], po1[0]), 3, (255,0, 0), 3)
            cv2.circle(proj_xform, (po2[1] + image1.shape[1], po2[0]), 3, (255,0, 0), 3)
    cv2.imshow('projective matching result', proj_xform)
    cv2.imwrite('projective_matching.png',proj_xform)
    return result

