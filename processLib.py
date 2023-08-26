"""
File: processLib.py

Data Processing Methods

Author: Nerissa Goh

Version:
0.1     20 Jul 23   new today
1.0     21 Jul 23   fixed start/stop key control (released as prelim product)

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import pickle
import mediapipe as mp
import math
import time
import landmarks as lm
import mask as m
# import imutils
import sys
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QGraphicsDropShadowEffect


def gaborFilter(image, theta, lambd, sigma, gamma, psi, kSize):
    '''
    The function returns gabor filtered image
    Parameters:
    image: Input image
    theta (float): orientation of sine function in radians
    lambd (float): wavelength od sine function
    sigma (float): standard deviation of Gaussian envelope
    gamma (float): aspect ratioof Gabor function
    psi: (float): phase offset of sine function
    kSize (tuple): kernel size

    Returns:
    filtered_image: Gabor filtered image
    '''
    try:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gabor_kernel = cv2.getGaborKernel(kSize, sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        # Filter the image using the Gabor kernel
        filtered_image = cv2.filter2D(gray, -1, gabor_kernel) # convolute image
        return filtered_image
    except Exception as e:
        print("An exception has occurred:", e)
        return image
    
    # # display kernel
    # # kernelVisual = np.uint8((gabor_kernel+1)/2 *255)
    # cv2.imshow('Kernel',cv2.resize(gabor_kernel,(400,400)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    
    # # Filter the image using the Gabor kernel
    # filtered_image = cv2.filter2D(gray, -1, gabor_kernel) # convolute image
    
    # return filtered_image
    
def getGaborFeatures(image, numTheta, numLambd, sigma, gamma, psi, kSize, targetSize, gaborFilter):
    '''
    This function returns the Gabor features of an input image
    Parameters:
    image: Input image
    numTheta (int): number of thetas in bank
    numLambd (int): number of lambd in bank
    sigma (float): standard deviation of Gaussian envelope
    gamma (float): aspect ratioof Gabor function
    psi: (float): phase offset of sine function
    kSize (tuple): kernel size

    Returns:
    Gabor features (np.array)
    '''
    numImgPerFeature = numLambd*numTheta
    gaborImages = []
    lambdVal = [4,8]
    
    for i in range(numLambd):
        for j in range(numTheta):
            theta = j*(np.pi/numTheta)
            lambd = lambdVal[i]
            filtered_image = gaborFilter(cv2.resize(image,(targetSize,targetSize)), theta, lambd, sigma=sigma, gamma=gamma, psi=psi, kSize=kSize)
            gaborImages.append(np.array(filtered_image).reshape(targetSize*targetSize))
            
            # # display gabor filtered images
            # # kernelVisual = np.uint8((gabor_kernel+1)/2 *255)
            # cv2.imshow('Gabor',filtered_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # cv2.waitKey(1)
            
    return np.array(gaborImages).reshape(numImgPerFeature*targetSize*targetSize)

def getCroppedFace(annotated_image_4crop, thres, face_landmarks, face_detection, targetSize, faceDetected):
    '''
    This function returns a face-cropped image
    Parameters:
    image: Input image
    thres (int): minimum number of pixel from tip of nose to face outline edge to select non-frontal face outline
    face_landmarks (obj): face landmarks
    face_detection (obj): face detection
    targetSize (int): target width and length of cropped image
  

    Returns:
    annotated_image_cropped, bbox_xmin, bbox_ymin, bbox_width, bbox_height
    '''
    faceLinesOrdered = []
    
    # determine the correct faceoutline
    # left profile?
    if abs(face_landmarks.landmark[323].x*annotated_image_4crop.shape[1] - face_landmarks.landmark[4].x*annotated_image_4crop.shape[1]) < thres or \
        abs(face_landmarks.landmark[361].x*annotated_image_4crop.shape[1] - face_landmarks.landmark[4].x*annotated_image_4crop.shape[1]) < thres:
        
        for left in lm.leftFaceOutlineArray:
            cx_ = int(face_landmarks.landmark[left].x*annotated_image_4crop.shape[1])
            cy_ = int(face_landmarks.landmark[left].y*annotated_image_4crop.shape[0])
            faceLinesOrdered.append([cx_,cy_])
        
    # right profile?
    elif abs(face_landmarks.landmark[4].x*annotated_image_4crop.shape[1] - face_landmarks.landmark[132].x*annotated_image_4crop.shape[1]) < thres or \
        abs(face_landmarks.landmark[4].x*annotated_image_4crop.shape[1] - face_landmarks.landmark[93].x*annotated_image_4crop.shape[1]) < thres:

        for right in lm.rightFaceOutlineArray:
            cx_ = int(face_landmarks.landmark[right].x*annotated_image_4crop.shape[1])
            cy_ = int(face_landmarks.landmark[right].y*annotated_image_4crop.shape[0])
            faceLinesOrdered.append([cx_,cy_])
    
    # frontal?
    else:
        for p in lm.faceOutlineArray:
            cx_ = int(face_landmarks.landmark[p].x*annotated_image_4crop.shape[1])
            cy_ = int(face_landmarks.landmark[p].y*annotated_image_4crop.shape[0])
            faceLinesOrdered.append([cx_,cy_])

    mask = np.zeros((annotated_image_4crop.shape[0], annotated_image_4crop.shape[1]))
    mask = cv2.fillConvexPoly(mask, np.array(faceLinesOrdered), 255)
    mask = mask.astype(bool)
    out = np.zeros_like(annotated_image_4crop)
    out[mask] = annotated_image_4crop[mask]
    annotated_image_outlined = out

    # detect face
    faceResults = face_detection.process(annotated_image_4crop)
    
    if faceResults.detections:
        detection = faceResults.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, ch = annotated_image_4crop.shape
        bbox_xmin = int(bbox.xmin * w)
        bbox_ymin = int(bbox.ymin * h)
        bbox_width = int(bbox.width * w)
        bbox_height = int(bbox.height * h)
    
        # crop face and make it into a square
        annotated_image_cropped = annotated_image_outlined[bbox_ymin:bbox_ymin+bbox_height, bbox_xmin:bbox_xmin+bbox_width]
        size = min(annotated_image_cropped.shape[0], annotated_image_cropped.shape[1])
        annotated_image_cropped = annotated_image_cropped[0:size, 0:size]
        
        try:
            if faceDetected and size > 0:
                # resize image to target size
                annotated_image_cropped = cv2.resize(annotated_image_cropped, (targetSize, targetSize))
                return annotated_image_cropped, bbox_xmin, bbox_ymin, bbox_width, bbox_height
        except Exception as e:
            print("Exception: annotated_image_cropped", e)

def minMaxScale(params, sample):
    '''
    This function returns a minmax scaled array of input feature vector
    Parameters:
    params: min/max scaling parameters 
    sample: Input feature vector

    Returns:
    minmax scaled np.array of input feature vector
    '''
    sampleScaled = []
    dataMin = params['minValues']
    dataMax = params['maxValues']

    for i, (minVal, maxVal) in enumerate(zip(dataMin, dataMax)):
        sampleScaled.append((sample[0][i]-minVal)/(maxVal-minVal))
        
    return np.array(sampleScaled).reshape(1,-1)