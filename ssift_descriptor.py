import cv2
import numpy as np
import matplotlib.pyplot as plt

def ssift_descriptor(feature_coords,image):
    """
    Computer Vision 600.461/661 Assignment 2
    Args:
        feature_coords (list of tuples): list of (row,col) tuple feature coordinates from image
        image (numpy.ndarray): The input image to compute ssift descriptors on. Note: this is NOT the image name or image path.
    Returns:
        descriptors (dictionary{(row,col): 128 dimensional list}): the keys are the feature coordinates (row,col) tuple and
                                                                   the values are the 128 dimensional ssift feature descriptors.
    """

    def normalize(feature):
        sums = 0
        for i in xrange(len(feature)):
            sums += feature[i] ** 2
        sums = np.sqrt(sums)
        feature = feature / sums
        return feature

    descriptors = dict()
    ix = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    iy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    kernel = cv2.getGaussianKernel(3, 1)
    image = image.astype(np.float64)
    image = cv2.filter2D(image, -1, kernel)
    image = cv2.filter2D(image, -1, kernel.T)
    g_x = cv2.filter2D(image, -1, ix)
    g_y = cv2.filter2D(image, -1, iy)
    mg=np.sqrt(g_x**2+g_y**2)
    angle=np.arctan2(g_x,g_y)
    angle=angle+np.pi*2
    for i in xrange(len(feature_coords)):
        shape=20
        pos=feature_coords[i]
        descriptors[pos[0],pos[1]]=[]
        if pos[0] < shape or pos[0] > image.shape[0] - shape or pos[1] < shape or pos[1] > image.shape[
            1] - shape or pos[0] < shape or pos[0] > image.shape[0] - shape or pos[1] < shape or pos[1] > \
                        image.shape[1] - shape:
            descriptors[pos[0],pos[1]]=[-1]
            continue
        mg1 = mg[pos[0] - shape:pos[0] + shape, pos[1] - shape:pos[1] + shape]
        mg1 = mg1.copy()
        gk=cv2.getGaussianKernel(40,1)
        mg1=cv2.filter2D(mg1,-1,gk)
        mg1 = cv2.filter2D(mg1, -1, gk.T)
        an1=angle[pos[0] - shape:pos[0] + shape, pos[1] - shape:pos[1] + shape]
        an1 = an1.copy()
        smg1 = np.hsplit(mg1, 4)
        for j in xrange(len(smg1)):
            smg1[j] = np.vsplit(smg1[j], 4)
        san1 = np.hsplit(an1, 4)
        for j in xrange(len(san1)):
            san1[j] = np.vsplit(san1[j], 4)
        b=[]
        for k in xrange(4):
            for q in xrange(4):
                a = [0]*8
                for z in xrange(10):
                    for x in xrange(10):
                        index=int(round(san1[k][q][z][x]*4/np.pi)%8)
                        a[index]+=smg1[k][q][z][x]
                b.extend(a)
        b=normalize(b)
        for w in xrange(len(b)):
            b[w]=b[w] if b[w]<0.2 else 0.2
        b=normalize(b)
        descriptors[pos[0],pos[1]]=b
    return descriptors

