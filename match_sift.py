import cv2
import numpy as np
import ssift_descriptor
import matplotlib.pyplot as plt

def match_features(feature_coords1,feature_coords2,f1,f2):
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
    for i in xrange(len(feature_coords1)):
        dis1=[]
        if len(f1[feature_coords1[i][0],feature_coords1[i][1]])==1:
            continue
        for j in xrange(len(feature_coords2)):
            if len(f2[feature_coords2[j][0], feature_coords2[j][1]]) == 1:
                continue
            dis1.append((compare(feature_coords1[i],feature_coords2[j],f1,f2),i,j))
        dis1.sort()
        match1=dis1[0][0]/(dis1[1][0]+0.000000001)
        if match1<0.7:
            matches.append((dis1[0][1],dis1[0][2]))
    return matches

def compare(pos1,pos2,f1,f2):
    a=np.array(f1[pos1[0],pos1[1]])
    b = np.array(f2[pos2[0], pos2[1]])
    c=(a-b)**2
    n=np.sqrt(c)
    sum=np.sum(n)
    return sum