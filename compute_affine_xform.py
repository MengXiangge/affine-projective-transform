# Author: TK
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_affine_xform(matches, features1, features2, image1, image2):
    affine_xform = np.zeros((3, 3))
    a = np.zeros((6,6), np.float64)
    a1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    a2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    h = np.zeros((12000, 3, 3), np.float64)
    q = np.zeros((12000, 1), np.float64)
    b=np.zeros((6,1), np.float64)
    inlier = list()
    if len(matches)<4:
        print 'Too few matchs to calculate affine transform'
        proj_xform = np.hstack((a1, a2))
        for i in xrange(len(matches)):
            po1 = features1[matches[i][0]]
            po2 = features2[matches[i][1]]
            cv2.line(proj_xform, (po1[1], po1[0]), (po2[1] + image1.shape[1], po2[0]), (255, 0,0), 1)
            cv2.circle(proj_xform, (po1[1], po1[0]), 3, (255, 0,0), 3)
            cv2.circle(proj_xform, (po2[1] + image1.shape[1], po2[0]), 3, (255, 0,0), 3)
        cv2.imshow('affine matching result', proj_xform)
        cv2.imwrite('affine_matching.png', proj_xform)
        return np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    for i in xrange(12000):
        inlier.append([])
        index = np.random.random_integers(0, len(matches) - 1, 3)
        while (len(set(index)) < len(index)):
            index = np.random.random_integers(0, len(matches) - 1, 3)
        for j in xrange(3):
            po1 = features1[matches[index[j]][0]]
            po2 = features2[matches[index[j]][1]]
            a[2 * j] = [po1[1], po1[0], 1, 0, 0, 0]
            a[2 * j + 1] = [0, 0, 0, po1[1], po1[0], 1]
            b[2*j]=po2[1]
            b[2*j+1]=po2[0]
        at=a.T
        try:
            t=np.linalg.inv(at.dot(a))
        except:
            continue
        else:
            t=t.dot(at)
            t=t.dot(b)
            last=np.array([0,0,1])
            r = np.reshape(t, (2, 3))
            r=np.vstack((r,last))
            h[i] = r
            for k in xrange(len(matches)):
                po1 = features1[matches[k][0]]
                po2 = features2[matches[k][1]]
                x1 = np.array([po1[1], po1[0], 1])
                x2 = np.dot(x1, r)
                if np.sqrt((x2[1] - po2[0]) ** 2 + (x2[0] - po2[1]) ** 2) <20:
                    q[i] += 1
                    inlier[i].append(k)
    i = np.where(q == max(q))
    inl = inlier[i[0][0]]
    result = h[i[0][0]]
    proj_xform = np.hstack((image1, image2))
    for i in xrange(len(matches)):
        po1 = features1[matches[i][0]]
        po2 = features2[matches[i][1]]
        if i in inl:
            cv2.line(proj_xform, (po1[1], po1[0]), (po2[1] + image1.shape[1], po2[0]), (0, 255, 0), 1)
            cv2.circle(proj_xform, (po1[1], po1[0]), 3, (0, 255, 0), 3)
            cv2.circle(proj_xform, (po2[1] + image1.shape[1], po2[0]), 3, (0, 255, 0), 3)
        else:
            cv2.line(proj_xform, (po1[1], po1[0]), (po2[1] + image1.shape[1], po2[0]), (255, 0, 0), 1)
            cv2.circle(proj_xform, (po1[1], po1[0]), 3, (255, 0, 0), 3)
            cv2.circle(proj_xform, (po2[1] + image1.shape[1], po2[0]), 3, (255, 0, 0), 3)
    cv2.imshow('affine matching result', proj_xform)
    cv2.imwrite('affine_matching.png', proj_xform)
    if len(inl)<4:
        return result
    else:
        res = list()
        for i in xrange(20*len(inl)**2):
            index = np.random.random_integers(0, len(inl) - 1, 3)
            while(len(set(index))<len(index)):
                index = np.random.random_integers(0, len(inl) - 1, 3)
            for j in xrange(3):
                po1 = features1[matches[inl[index[j]]][0]]
                po2 = features2[matches[inl[index[j]]][1]]
                a[2 * j] = [po1[1], po1[0], 1, 0, 0, 0]
                a[2 * j + 1] = [0, 0, 0, po1[1], po1[0], 1]
                b[2 * j] = po2[1]
                b[2 * j + 1] = po2[0]
            at = a.T
            t = np.linalg.inv(at.dot(a))
            t = t.dot(at)
            t = t.dot(b)
            t=np.ndarray.tolist(t)
            res.append(t)
        res=np.array(res)
        w=np.mean(res,axis=0)
        last = np.array([0, 0, 1])
        r = np.reshape(w, (2, 3))
        r = np.vstack((r, last))
        return r

