import numpy as np
def merge(img1,img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    rows= max(img1.shape[0],img2.shape[0])
    cols = max(img1.shape[1], img2.shape[1])
    result=np.zeros((rows,cols,3),np.float64)
    for i in xrange(img1.shape[0]):
        for j in xrange(img1.shape[1]):
            result[i][j]+=img2[i][j]*0.5
    for i in xrange(img1.shape[0]):
        for j in xrange(img1.shape[1]):
            result[i][j]+=0.5*img1[i][j]
    result = result.astype(np.uint8)
    return result