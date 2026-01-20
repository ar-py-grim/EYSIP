#!/usr/bin/env python3

import cv2
import numpy as np


def aruco_detect_edge(image):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)  
    parameters = cv2.aruco.DetectorParameters()
    if image is not None:
        corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)
        return corners,ids

def aruco_detect_bot(image):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)  
    parameters = cv2.aruco.DetectorParameters()
    if image is not None:
        corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)
        return corners,ids
    
def aruco_detect1(image):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)  
    parameters = cv2.aruco.DetectorParameters()
    if image is not None:
        corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)
        return corners,ids  


def camera_caliberation(distorted_image):
    camera_matrix = np.array([[1959.122975, 0.0, 629.170243],
                                       [0.0, 2007.411302, 325.922826],
                                       [0.0, 0.0, 1.0]])

    dist_coefficients = np.array([-2.887594, 6.042950, -0.030759, -0.006263, 0.000000])
    if distorted_image is not None:
        cv_image = cv2.undistort(distorted_image, camera_matrix, dist_coefficients)
        return cv_image

def thresholding(image):
    if image is not None:
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(image, 120, 230, cv2.THRESH_BINARY)
        return thresholded

def get_corners(corners,ids):
    topleft=None
    topright=None
    bottomleft=None
    bottomright=None
    all_id=[85,90,80,95]
    for i in range(len(corners)):
        if ids[i][0]==all_id[0]:bottomleft=corners[i][0][3]
            
        if ids[i][0]==all_id[1] :bottomright=corners[i][0][2]
        if ids[i][0]==all_id[2]:topright=corners[i][0][1]
        if ids[i][0]==all_id[3]:topleft=corners[i][0][0]
    

    return topleft,topright,bottomleft,bottomright
    
def perspective_transform(image,arena_corners):
    req_corners=np.float32([[0,0],[500,0],[0,500],[500,500]])
    matrix=cv2.getPerspectiveTransform(arena_corners,req_corners)
    final_image=cv2.warpPerspective(image,matrix,(500,500))

    return final_image