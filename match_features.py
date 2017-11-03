import cv2
import numpy as np
import matplotlib.pyplot as plt

def match_features(feature_coords1,feature_coords2,image1,image2):
    """
    Computer Vision 600.461/661 Assignment 2
    Args:
        feature_coords1 (list of tuples): list of (row,col) tuple feature coordinates from image1
        feature_coords2 (list of tuples): list of (row,col) tuple feature coordinates from image2
        image1 (numpy.ndarray): The input image corresponding to features_coords1
        image2 (numpy.ndarray): The input image corresponding to features_coords2
    Returns:
        matches (list of tuples): list of index pairs of possible matches. For example, if the 4-th feature in feature_coords1 and the 0-th feature
                                  in feature_coords2 are determined to be matches, the list should contain (4,0).
    """
    matches = list()
    matches1 = list()
    matches2 = list()
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)
    for i in xrange(len(feature_coords1)):
        match1=-1
        po1=0
        for j in xrange(len(feature_coords2)):
            dis1=compare(feature_coords1[i],feature_coords2[j],image1,image2)
            if match1<dis1:
                po1=j
                match1=dis1
        if match1<0.2:
            matches1.append(-1)
        else:
            matches1.append(po1)

    for i in xrange(len(feature_coords2)):
        match2=-1
        po2=0
        for j in xrange(len(feature_coords1)):
            dis2=compare(feature_coords2[i],feature_coords1[j],image2,image1)
            if match2<dis2:
                po2=j
                match2=dis2
        if match2<0.2:
            matches2.append(-1)
        else:
            matches2.append(po2)

    for i in xrange(len(feature_coords1)):
        if matches2[matches1[i]]==i:
            matches.append((i,matches1[i]))
    return matches

def compare(pos1,pos2,image1,image2):
    shape=15
    if pos1[0]<shape or pos1[0]>image1.shape[0]-shape or pos1[1]<shape or pos1[1]>image1.shape[1]-shape or pos2[0]<shape or pos2[0]>image2.shape[0]-shape or pos2[1]<shape or pos2[1]>image2.shape[1]-shape:
        return -2
    temp1=image1[pos1[0]-shape:pos1[0]+shape, pos1[1]-shape:pos1[1]+shape]
    temp1=temp1.copy()
    temp2 = image2[pos2[0] - shape:pos2[0] + shape, pos2[1] - shape:pos2[1] + shape]
    temp2 = temp2.copy()
    mu1=np.mean(temp1)
    mu2=np.mean(temp2)
    delta1=np.std(temp1)
    delta2=np.std(temp2)
    temp1=(temp1-mu1)/delta1
    temp2=(temp2-mu2)/delta2
    sum=np.mean(temp1*temp2)
    #for i in xrange(shape[0]):
    #    for j in xrange(shape[1]):
    #        a0=pos1[0]+i-(shape[0]-1)/2
    #        b0=pos1[1]+j-(shape[1]-1)/2
    #        a1 = pos2[0] +i-(shape[0]-1)/2
    #        b1 = pos2[1]+j-(shape[1]-1)/2
    #        tep1=image1[pos1[0]][pos1[1]] if a0==-1 or a0==image1.shape[0] or b0==-1 or b0==image1.shape[1] else image1[a0][b0]
    #        tep2 = image2[pos2[0]][pos2[1]] if a1 == -1 or a1 == image2.shape[0] or b1 == -1 or b1 == image2.shape[1] else image2[a1][b1]
    #        sum=sum+(tep1-tep2)**2
    return sum