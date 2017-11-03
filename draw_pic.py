import cv2
import detect_features
import match_features
import ssift_descriptor
import match_sift
import compute_affine_xform_sift
import compute_proj_xform
import compute_affine_xform
import merge
import matplotlib.pyplot as plt
def draw_pic(image1,image2):
    img1=cv2.imread(image1)
    img2=cv2.imread(image2)
    img1=cv2.resize(img1,(img2.shape[1],img2.shape[0]))
    a1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    b1=detect_features.detect_features(a1)
    a2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    b2=detect_features.detect_features(a2)
    rows,cols,ch = img2.shape
    # m=match_features.match_features(b1,b2,a1,a2)
    # a_result=compute_affine_xform.compute_affine_xform(m,b1,b2,img1,img2)
    # p_result=compute_proj_xform.compute_proj_xform(m,b1,b2,img1,img2)
    # a_res=cv2.warpPerspective(img1,a_result,(cols,rows))
    #a_r=merge.merge(a_res,img2)
    # a_r=cv2.addWeighted(img2,0.5,a_res,0.5,0.0)
    # cv2.imshow('ZCC affine match result', a_r)
    # cv2.imwrite('ZCC_affine_match_result.png', a_r)
    # p_res = cv2.warpPerspective(img1, p_result, (cols, rows))
    # # r=merge.merge(res,img2)
    # p_r = merge.merge(p_res, img2)
    #p_r = cv2.addWeighted(img2, 0.5, p_res, 0.5, 0.0)
    # cv2.imshow('projective match result', p_r)
    # cv2.imwrite('projective_match_result.png', p_r)
    f1 = ssift_descriptor.ssift_descriptor(b1, a1)
    f2 = ssift_descriptor.ssift_descriptor(b2, a2)
    sift_m = match_sift.match_features(b1, b2, f1, f2)
    sift_a_result = compute_affine_xform_sift.compute_affine_xform_sift(sift_m, b1, b2, img1, img2)
    sift_res = cv2.warpPerspective(img1, sift_a_result, (cols, rows))
    # r=merge.merge(res,img2)
    sift_r = merge.merge(sift_res, img2)
    #sift_r = cv2.addWeighted(img2, 0.5, sift_res, 0.5, 0.0)
    cv2.imshow('sift affine match result', sift_r)
    cv2.imwrite('affine_sift_match_result.png', sift_r)
    cv2.waitKey()
    cv2.destroyAllWindows()