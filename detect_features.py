import cv2
import numpy as np
import matplotlib.pyplot as plt
import nonmaxsuppts

def detect_features(image):
    """
    Computer Vision 600.461/661 Assignment 2
    Args:
        image (numpy.ndarray): The input image to detect features on. Note: this is NOT the image name or image path.
    Returns:
        pixel_coords (list of tuples): A list of (row,col) tuples of detected feature locations in the image
    """
    pixel_coords = list()
    ix=np.array([[-1,-2,0,2,1],[-2,-3,0,3,2],[-3,-5,0,5,3],[-2,-3,0,3,2],[-1,2,0,2,1]])
    iy=np.array([[1,2,3,2,1],[2,3,5,3,2],[0,0,0,0,0],[-2,-3,-5,-3,-2],[-1,-2,-3,-2,-1]])
    kernel=cv2.getGaussianKernel(5,1)
    image = image.astype(np.float64)
    image= cv2.filter2D(image, -1, kernel)
    image = cv2.filter2D(image, -1, kernel.T)
    g_x=cv2.filter2D(image,-1,ix)
    g_y=cv2.filter2D(image,-1,iy)
    ix2=np.multiply(g_x,g_x)
    iy2 = np.multiply(g_y, g_y)
    ixy=np.multiply(g_x, g_y)
    kernel = cv2.getGaussianKernel(7, 2)
    gu_k=np.dot(kernel,kernel.T)
    ix2=cv2.filter2D(ix2, -1, gu_k)
    iy2=cv2.filter2D(iy2, -1, gu_k)
    ixy=cv2.filter2D(ixy, -1, gu_k)
    corner=np.zeros(image.shape, np.float64)
    k=0.05
    for i in xrange(image.shape[0]):
        for j in xrange(image.shape[1]):
            det=ix2[i][j]*iy2[i][j]-1/4*ixy[i][j]*ixy[i][j]
            trace=ix2[i][j]+iy2[i][j]
            corner[i][j]=det-k*trace*trace
    a=int(corner.max()/10)
    pixel_coords=nonmaxsuppts.nonmaxsuppts(corner,7,0.2*a)
    return pixel_coords
