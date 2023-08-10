"""
File: truthMirrorFP2.py

Class MainWindow

Main application window for Face Emotion Recognition

Author: Nerissa Goh

Version:
0.1     4 Jul 23    new today
0.2     10 Jul 23   ported ipynb functionality to py
0.3     12 Jul 23   completed update_frame() method
0.4     15 Jul 23   incorporated UI using PyQt5
0.5     17 Jul 23   Completed button click functions
0.6     18 Jul 23   Keyboard buttons functional
0.7     19 Jul 23   Fine-tuned layout
0.8     20 Jul 23   breakout button class and data processing methods into separate py files
1.0     21 Jul 23   fixed start/stop key control (released as prelim product)
1.1     25 Jul 23   Error Handling
2.0     30 Jul 23   Gesture left right navigation
2.1     2 Aug 23    Refactoring

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
# import mask as m
import sys
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QGraphicsDropShadowEffect
import processLib as pl
import controlLib as cl
from collections import deque

class MainWindow(QWidget): #MainWindow class inheriting from QWidget
    def __init__(self):
        super().__init__()
        
        ##### variable settings
        self.delayTime = 0.4 #secs
        self.prevTime = time.time() # previous time of emotion prediction display update
        self.baseIPD = 500 # standardized interpupiliary distance
        self.thres = 10 # face turn in pixels 
        self.targetSize = 96 # image size
        self.text_0 = '' # ipd text
        self.text_1 = '' # ipd scaling text
        self.text2 ='' # emotion text
        self.imageTemplateNum = 0 #current template image number
        self.tabNum = 0 # current tab position
        self.vidOn = True # video status bit
        self.faceMeshOn = False # control face mesh display
        self.emo = ['angry', 
                    'disgust',
                    'fear',
                    'happy',
                    'sad',
                    'surprised',
                    'neutral'
                    ] # emotion text
        self.imageTemplateName = [
                        ['happy1', 'happy2'],
                        ['surprised1', 'surprised2'],
                        ['angry1', 'angry2'],
                        ['fear1', 'fear2'],
                        ['sad1', 'sad2'],
                        ['disgust1', 'disgust2'],
                        ['neutral1', 'neutral2'],
                    ] # template image filenames
        self.imageTitles = [
                        ['Happy Face 1', 'Happy Face 2'],
                        ['Surprised Face 1', 'Surprised Face 2'],
                        ['Angry Face 1', 'Angry Face 2'],
                        ['Fearful Face 1', 'Fearful Face 2'],
                        ['Sad Face 1', 'Sad Face 2'],
                        ['Disgusted Face 1', 'Disgusted Face 2'],
                        ['Neutral Face 1', 'Neutral Face 2'],
                    ] # template iamge titles
        
        self.readDataPath = './resources/'
        self.missingResource = False # model files missing status
        self.faceDetected = False # face detected status
        self.indexThumbSep = 40 # pitch distance
        self.leftOn = 0 # left click
        self.rightOn = 0 # right click
        self.frameCount = 0 # image frame count
        self.index = (0,0) # index tip coordinate
        self.thumb = (0,0) # thumb tip coordinate
        self.sampleHand = 4 # number of frames per hand sample
        self.fpsFifo = deque([0] * self.sampleHand, maxlen=self.sampleHand) # init fps fifo to all 0s
        
        ##### Read App Resources
        try:
            # load SVM predictive model
            with open(self.readDataPath+'combiModels/modelSvmRbf_pp.pkl', 'rb') as file:
                self.model = pickle.load(file)
        except Exception as e:
            self.missingResource = True
            print(f"{self.readDataPath+'combiModels/modelSvmRbf_pp.pkl'} does not exist",e)

        # 3D landmarks features
        try:
            # load minmaxscaler parameters
            with open(self.readDataPath+'combiModels/scalerParams_svm_pp.pkl', 'rb') as file:
                self.params = pickle.load(file)
        except Exception as e:
            self.missingResource = True
            print(f"{self.readDataPath+'combiModels/scalerParams_svm_pp.pkl'} does not exist",e)
        
        try:
            # load trained pca model
            with open(self.readDataPath+'combiModels/pca_svm_pp.pkl', 'rb') as file:
                self.pca = pickle.load(file)
        except Exception as e:
            self.missingResource = True
            print(f"{self.readDataPath+'combiModels/pca_svm_pp.pkl'} does not exist",e)
               
        # Gabor features 
        try: 
            # load minmaxscaler parameters
            with open(self.readDataPath+'combiModels/scalerParamsGabor_svm_pp.pkl', 'rb') as file:
                self.paramsGabor = pickle.load(file)
        except Exception as e:
            self.missingResource = True
            print(f"{self.readDataPath+'combiModels/scalerParamsGabor_svm_pp.pkl'} does not exist",e)
        
        try:
        # load trained pca model
            with open(self.readDataPath+'combiModels/pca_svm_gabor_pp.pkl', 'rb') as file:
                self.pcaGabor = pickle.load(file) 
        except Exception as e:
            self.missingResource = True
            print(f"{self.readDataPath+'combiModels/pca_svm_gabor_pp.pkl'} does not exist",e)
        
        ##### instaniate Mediapipe objects
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh

        # face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.3)
        
        # MediaPipe HandLandmark model
        self.mp_hands = mp.solutions.hands

        ###### create the video capture object
        self.cap = cv2.VideoCapture(0)

        ###### set up a timer to capture frames periodically
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30 milliseconds (33 frames per second)
        
        self.initUI()

    def initUI(self):
        self.setWindowTitle(f"Truthful Mirror Ver 2.1")
        self.resize(1400, 600)
        self.window = QWidget()
        self.setStyleSheet("background-color: #bfbeb6;") # set background color

        ######## Widgets
        # create labels to display the video stream and images
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.leftImage_label = QLabel(self)
        self.leftImage_label.setAlignment(Qt.AlignCenter)
        self.rightImage_label = QLabel(self)
        self.rightImage_label.setAlignment(Qt.AlignCenter)
        
        # Create a QGraphicsDropShadowEffect for the video label
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)  # blur radius to control the shadow's blur level
        shadow.setXOffset(5)      # X offset to control the shadow's horizontal position
        shadow.setYOffset(5)      # Y offset to control the shadow's vertical position

        # Apply the shadow effect to each label
        self.video_label.setGraphicsEffect(shadow)
        
        self.leftImageTitle_label = QLabel(f"{self.imageTitles[self.imageTemplateNum][0]}", self)
        self.font = QFont('Arial', 24)  # set font and size
        self.font.setWeight(QFont.Bold)  # set font weight to bold for thicker text
        self.leftImageTitle_label.setFont(self.font)
        self.leftImageTitle_label.setAlignment(Qt.AlignCenter)
        self.leftImageTitle_label.setStyleSheet("color: black;")
        
        self.mirrorTitle_label = QLabel("Your Face",self)
        self.font = QFont('Arial', 24)  # set font and size
        self.font.setWeight(QFont.Bold)  # set font weight to bold for thicker text
        self.mirrorTitle_label.setFont(self.font)
        self.mirrorTitle_label.setAlignment(Qt.AlignCenter)
        self.mirrorTitle_label.setStyleSheet("color: black;")
        
        self.rightImageTitle_label = QLabel(f"{self.imageTitles[self.imageTemplateNum][1]}", self)
        self.font = QFont('Arial', 24)  # Replace 'Arial' with your desired font family and 20 with the desired font size
        self.font.setWeight(QFont.Bold)  # Set font weight to bold for thicker text
        self.rightImageTitle_label.setFont(self.font)
        self.rightImageTitle_label.setAlignment(Qt.AlignCenter)
        self.rightImageTitle_label.setStyleSheet("color: black;")
         
        ###### create pixelmaps to display template images
        self.blackImage = np.zeros((393, 290, 3), dtype=np.uint8) # blank image if image does not exist
        # overlay text
        cv2.putText(self.blackImage, 'Image not', ((self.blackImage.shape[1] // 2)-75, self.blackImage.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
        cv2.putText(self.blackImage, 'Found', ((self.blackImage.shape[1] // 2)-40, (self.blackImage.shape[0] // 2)+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
        
        ##### update template images
        cl.updateTemplateImages(self.imageTemplateName, self.imageTemplateNum, self.blackImage, self.leftImage_label, self.rightImage_label)
        
        # Create a QGraphicsDropShadowEffect for the label
        shadowL = QGraphicsDropShadowEffect()
        shadowL.setBlurRadius(10)  # blur radius to control the shadow's blur level
        shadowL.setXOffset(5)      # X offset to control the shadow's horizontal position
        shadowL.setYOffset(5)      # Y offset to control the shadow's vertical position
        
        # Apply the shadow effect to each label
        self.leftImage_label.setGraphicsEffect(shadowL)
        
        # Create a QGraphicsDropShadowEffect for the label
        shadowR = QGraphicsDropShadowEffect()
        shadowR.setBlurRadius(10)  # blur radius to control the shadow's blur level
        shadowR.setXOffset(5)      # X offset to control the shadow's horizontal position
        shadowR.setYOffset(5)      # Y offset to control the shadow's vertical position
        
        # Apply the shadow effect to each label
        self.rightImage_label.setGraphicsEffect(shadowR)
        
        ###### create buttons for controls
        self.start_button = cl.CreateButton("Start Video")
        self.stop_button = cl.CreateButton("Stop Video")
        self.mesh_button = cl.CreateButton("Face Mesh")
        self.left_button = cl.CreateButton("Previous")
        self.right_button = cl.CreateButton("Next")
        self.right_button.setStyleSheet( # next button highlighted
            "background-color: #dbab30;"
            "border: 5px solid red;"
            "border-radius: 20px;"
            "color: black;"
            "padding: 10px 20px;"
        )
        
        ####### Layout
        # create a horizontal layout for text titles for images and video
        self.title_layout = QHBoxLayout()
        self.title_layout.addWidget(self.leftImageTitle_label)
        self.title_layout.addWidget(self.mirrorTitle_label)
        self.mirrorTitle_label.setFixedWidth(900)
        self.title_layout.addWidget(self.rightImageTitle_label)
        
        # create a horizontal layout for images & video screen
        self.emo_layout = QHBoxLayout()
        self.emo_layout.addWidget(self.leftImage_label)
        self.leftImage_label.setFixedWidth(200)
        self.leftImage_label.setFixedHeight(393)
        self.emo_layout.addWidget(self.video_label)
        self.video_label.setFixedWidth(900)
        self.video_label.setFixedHeight(506)
        self.emo_layout.addWidget(self.rightImage_label)
        self.rightImage_label.setFixedWidth(200)
        self.rightImage_label.setFixedHeight(393)
    
        # create a horizontal layout for control buttons
        self.cntl_layout = QHBoxLayout()
        self.cntl_layout.addWidget(self.start_button)
        self.cntl_layout.addWidget(self.mesh_button)
        self.cntl_layout.addWidget(self.stop_button)
        
        # create a horizontal layout for navigation buttons
        self.nav_layout = QHBoxLayout()
        self.nav_layout.addWidget(self.left_button)
        self.nav_layout.addWidget(self.right_button)
        
        # create a vertical layout to arrange the widgets
        self.layout = QVBoxLayout()
        self.layout.addLayout(self.title_layout)
        self.layout.addLayout(self.emo_layout)
        self.layout.addLayout(self.cntl_layout)
        self.layout.addLayout(self.nav_layout)
        
        # set the layout for the widget
        self.setLayout(self.layout)
        
        ###### set button click event handlers
        self.start_button.clicked.connect(self.start_video)
        self.stop_button.clicked.connect(self.stop_video)
        self.left_button.clicked.connect(self.change_to_prev)
        self.right_button.clicked.connect(self.change_to_next)
        self.mesh_button.clicked.connect(self.mesh_toggle)

    def start_video(self):
        '''
        This function starts video pipe
        Parameters:
        None

        Returns:
        Nothing
        '''
        # start capturing video frames
        self.timer.start()
        self.vidOn = True

    def stop_video(self):
        '''
        This function stops video pipe
        Parameters:
        None

        Returns:
        Nothing
        '''
        self.timer.stop()
        self.vidOn = False
        
    def change_to_prev(self):
        '''
        This function change the template images to the next
        Parameters:
        None

        Returns:
        Nothing
        '''
        if self.imageTemplateNum == 0:
            self.imageTemplateNum = 6
        else:
            self.imageTemplateNum -= 1
        
        ##### update template images
        cl.updateTemplateImages(self.imageTemplateName, self.imageTemplateNum, self.blackImage, self.leftImage_label, self.rightImage_label)
        
        ###### update left image title
        self.leftImageTitle_label.setText(f"{self.imageTitles[self.imageTemplateNum][0]}")
        
        ###### update right image title
        self.rightImageTitle_label.setText(f"{self.imageTitles[self.imageTemplateNum][1]}")
    
    def change_to_next(self):
        '''
        This function change the template images to the previous
        Parameters:
        None

        Returns:
        Nothing
        '''
        if self.imageTemplateNum == 6:
            self.imageTemplateNum = 0
        else:
            self.imageTemplateNum += 1
        
         ##### update template images
        cl.updateTemplateImages(self.imageTemplateName, self.imageTemplateNum, self.blackImage, self.leftImage_label, self.rightImage_label)
        
        # update left image title
        self.leftImageTitle_label.setText(f"{self.imageTitles[self.imageTemplateNum][0]}")
        
        # update right image title
        self.rightImageTitle_label.setText(f"{self.imageTitles[self.imageTemplateNum][1]}")
    
    def mesh_toggle(self):
        '''
        This function toggles face mesh display
        Parameters:
        None

        Returns:
        Nothing
        '''
        if self.faceMeshOn:
            self.faceMeshOn = False
        else:
            self.faceMeshOn = True
            
    def keyPressEvent(self, event):
        '''
        This inherited function decodes key-down events
        Parameters:
        event

        Returns:
        Nothing
        '''
        key = event.key()
        if key == 16777234:  # left arrow key
            self.change_to_prev()
        elif key == 16777236:  # right arrow key
            self.change_to_next()
        elif event.key() == Qt.Key_M:
            self.mesh_toggle()
        elif event.key() == Qt.Key_Space:
            if self.vidOn:
                self.stop_video()
                self.vidOn = False
            else:
                self.start_video()
                self.vidOn = True
        elif event.key() == Qt.Key_Tab:
            if self.tabNum < 4:
                self.tabNum += 1
            else:
                self.tabNum = 0
            cl.next_tab(self.tabNum, self.stop_button, self.right_button, self.left_button, self.start_button, self.mesh_button)
        elif event.key() == Qt.Key_Return:
            if self.tabNum == 0:
                self.change_to_next()
            elif self.tabNum == 1:
                self.change_to_prev()
            elif self.tabNum == 2:
                self.start_video()
            elif self.tabNum == 3:
                self.mesh_toggle()
            elif self.tabNum == 4:
                self.stop_video()
        elif event.key() == Qt.Key_Q:
            self.close()

    def update_frame(self):
        '''
        This inherited function that updates the videp pipe display
        Parameters:
        None

        Returns:
        Nothing
        '''
        # Capture a frame from the video stream
        success, img = self.cap.read()
        self.faceDetected = False # reset face detected
        startFrame = time.time()
        
        if success:
            with self.mp_face_mesh.FaceMesh(max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

                # Flip the image horizontally for a selfie-view display.
                image = cv2.flip(img.copy(),1)
                annotated_image = image.copy() # for face detection
                h, w, c = image.shape
                
                # To improve performance, optionally mark the image as not writeable to pass by reference.
                image.flags.writeable = False
    
                results = face_mesh.process(image)
                
                image.flags.writeable = True
                
                # face landmarks processing and display
                if results.multi_face_landmarks:
                    self.faceDetected = True
                    for face_landmarks in results.multi_face_landmarks:
                        imgMatrix = []
                        # Compute Interpupillary Distance (IPD)
                        leftPupilx = int(face_landmarks.landmark[386].x*image.shape[1])
                        leftPupily = int(face_landmarks.landmark[386].y*image.shape[0])
                        leftPupilz = int(face_landmarks.landmark[386].z*image.shape[1])

                        rightPupilx = int(face_landmarks.landmark[159].x*image.shape[1])
                        rightPupily = int(face_landmarks.landmark[159].y*image.shape[0])
                        rightPupilz = int(face_landmarks.landmark[159].z*image.shape[1])

                        # cv2.line(image, (leftPupilx, leftPupily), (rightPupilx, rightPupily), (0, 255, 255), thickness=2 )
                        # cv2.circle(image, (leftPupilx, leftPupily), 2, (0, 255, 255), -1)
                        # cv2.circle(image, (rightPupilx, rightPupily), 2, (0, 255, 255), -1)

                        ipd = math.sqrt((rightPupilx-leftPupilx)**2 + (rightPupily-leftPupily)**2 + (rightPupilz-leftPupilz)**2)

                        for selectedLandmark in lm.selectedLandmarks:
                            cx = int(face_landmarks.landmark[selectedLandmark[0]].x*image.shape[1])
                            cy = int(face_landmarks.landmark[selectedLandmark[0]].y*image.shape[0])
                            cz = int(face_landmarks.landmark[selectedLandmark[0]].z*image.shape[1])
                            cx2 = int(face_landmarks.landmark[selectedLandmark[1]].x*image.shape[1])
                            cy2 = int(face_landmarks.landmark[selectedLandmark[1]].y*image.shape[0])
                            cz2 = int(face_landmarks.landmark[selectedLandmark[1]].z*image.shape[1])

                            # cv2.line(image, (cx, cy), (cx2, cy2), (0, 255, 0), thickness=2)
                            # cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
                            # cv2.circle(image, (cx2, cy2), 5, (0, 255, 0), -1)
                            
                            # compute Euclidean distance between coordinates and scale it based on base IPD
                            imgMatrix.append(math.sqrt((cx2-cx)**2 + (cy2-cy)**2 + (cz2-cz)**2) * (self.baseIPD/ipd))

                        # 3D landmark features
                        # PCA dimensionality reduction
                        landmarkSample = np.array(imgMatrix).reshape((1,-1))
                    
                        # scale 3D landmark features
                        landmarkSample_scaled = pl.minMaxScale(self.params, landmarkSample)
                    
                        # Reduce dimensionality using PCA
                        reduced_sample_pca = self.pca.transform(landmarkSample_scaled)
                        
                        # Gabor feature extraction
                        # adaptive face cropping
                        try:
                            try:
                                croppedFaceResults = pl.getCroppedFace(annotated_image, self.thres, face_landmarks, self.face_detection, self.targetSize, faceDetected=self.faceDetected)
                                if croppedFaceResults != None:
                                    imageCropped, bbox_xmin, bbox_ymin, bbox_width, bbox_height = croppedFaceResults
                                else:
                                    bbox_width = 0
                                    bbox_height = 0
                                    self.faceDetected = False
                                
                            except Exception as e:
                                self.faceDetected = False
                                print("Exception: pl.getCroppedFace",e)
                        
                            if not (bbox_width <= 0 or bbox_height <= 0) and self.faceDetected: # check for face detected during cropping
                                # extract Gabor features
                                numTheta = 5
                                numLambd = 2
                                sigma = 1  # Standard deviation of the Gaussian envelope
                                gamma = 0.5  # Aspect ratio of Gaussian envelope
                                psi = 290*(np.pi/180)  # Phase offset of sinusoidal function
                                kSize = (3,3)
                                
                                try:
                                    imageCroppedGabor = pl.getGaborFeatures(imageCropped, numTheta, numLambd, sigma, gamma, psi, kSize, self.targetSize, pl.gaborFilter)
                                except Exception as e:
                                    self.faceDetected = False
                                    print("Exception: imageCroppedGabor",e)
                                
                                try:
                                    # perform PCA on gabor features
                                    imageCroppedGaborPCA = self.pcaGabor.transform(imageCroppedGabor.reshape((1,-1)))
                                except Exception as e:
                                    self.faceDetected = False
                                    print("Exception: pcaGabor.transform",e)
                                
                                try:
                                    # scale reduced gabor features
                                    imageCroppedGaborPCA_scaled = pl.minMaxScale(self.paramsGabor, imageCroppedGaborPCA)
                                except Exception as e:
                                    self.faceDetected = False
                                    print("Exception: minMaxScale",e)
                                
                                try:
                                    # concatenate landmark and gabor datasets
                                    landmarkGaborComb = np.concatenate((reduced_sample_pca, imageCroppedGaborPCA_scaled), axis=1)     
                                except Exception as e:
                                    self.faceDetected = False
                                    print("Exception: concatenate",e)
                                
                                try:
                                    # predict with svm model
                                    y_pred = self.model.predict(landmarkGaborComb)
                                    confidence = self.model.predict_proba(landmarkGaborComb)  
                                except Exception as e:
                                    self.faceDetected = False
                                    print("Exception: model",e)
                                
                                # Face mesh display
                                if self.faceMeshOn:
                                    self.mp_drawing.draw_landmarks(image, face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
                                    )
                                # else:
                                #     if self.faceDetected:
                                #         # bounding box
                                #         cv2.rectangle(image, (bbox_xmin, bbox_ymin), (bbox_xmin + bbox_width, bbox_ymin + bbox_height), (0, 255, 0), 2)     
                            
                            else:
                                self.faceDetected = False

                        except Exception as e:
                                    self.faceDetected = False
                                    print("Exception: Gabor processing",e)
                      
                else:
                    self.faceDetected = False
            
            if self.frameCount % self.sampleHand == 0: # sample for hands every 4 frames to increase performance
                ##### gesture clicking         
                with self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3) as hands:

                    # Process the image with MediaPipe HandLandmark model
                    results = hands.process(image)

                    if results.multi_hand_landmarks:
                        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                            # get the handedness information (left or right)
                            hand_label = handedness.classification[0].label
                            # Get the tips of the index and thumbs (landmarks 4 and 8 for index, and 4 and 12 for thumb)
                            self.thumb = (int(hand_landmarks.landmark[4].x * image.shape[1]),
                                        int(hand_landmarks.landmark[4].y * image.shape[0]))
                            self.index = (int(hand_landmarks.landmark[8].x * image.shape[1]),
                                        int(hand_landmarks.landmark[8].y * image.shape[0]))

                            # Draw circles at the tips of the index and thumbs
                            if hand_label == 'Left':
                                if math.sqrt((self.thumb[0]-self.index[0])**2 + (self.thumb[1]-self.index[1])**2) < self.indexThumbSep and self.index[0] < w/2 and self.leftOn==0:
                                    cv2.circle(image, self.index, 20, (0, 255, 0), -1)  # green circle for index tip
                                    self.change_to_prev()
                                    self.leftOn = 1
                                elif math.sqrt((self.thumb[0]-self.index[0])**2 + (self.thumb[1]-self.index[1])**2) < self.indexThumbSep and self.index[0] < w/2 and self.leftOn==1:
                                    cv2.circle(image, self.index, 20, (0, 255, 0), -1)
                                    self.leftOn = 1
                                else:
                                    self.leftOn = 0
                            if hand_label == 'Right':
                                if math.sqrt((self.thumb[0]-self.index[0])**2 + (self.thumb[1]-self.index[1])**2) < self.indexThumbSep and self.index[0] >= w/2 and self.rightOn==0:
                                    cv2.circle(image, self.index, 20, (0, 0, 255), -1)  # red circle for index tip 
                                    self.change_to_next()
                                    self.rightOn = 1
                                elif math.sqrt((self.thumb[0]-self.index[0])**2 + (self.thumb[1]-self.index[1])**2) < self.indexThumbSep and self.index[0] >= w/2 and self.rightOn==1:
                                    cv2.circle(image, self.index, 20, (0, 0, 255), -1)
                                    self.rightOn = 1
                                else:
                                    self.rightOn = 0
                
            ###### Warning and Emotion display  
            if self.missingResource:
                self.text_0 = f"Distance from Camera = Unknown"
                textMissRes = 'Error: Missing Model Files'
                # get the width and height of the warning text2 box
                (text2Width, text2Height) = cv2.getTextSize(textMissRes, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, thickness=3)[0]
                # text background highlight
                cv2.rectangle(image, (int((w/2)-330)-5,int(h/2)-text2Height-5), (int((w/2)-330+text2Width+5),int(h/2)+5), (70,70,70), cv2.FILLED)
                cv2.putText(image,textMissRes, (int((w/2)-330),int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)
            elif self.faceDetected:
                # emotion display rate control
                currentTime = time.time()
                if (currentTime - self.prevTime) > self.delayTime:
                    distCam = -0.23*ipd+83.02 # empirical equation
                    self.text_0 = f"Distance from Camera = {round(distCam if distCam >= 0 else 0 ,2)} cm"
                    self.text_1 = f"Scaling factor = {round((self.baseIPD/ipd),2)}"
                    self.text2 = f"{self.emo[y_pred[0]]} : {round(confidence[0][y_pred[0]]*100)}%"
                    # if confidence[0][y_pred[0]] >0.5:
                    #     self.text2 = f"{emo[y_pred[0]]} : {round(confidence[0][y_pred[0]]*100)}%"
                    #     # text2 = f"{emo[y_pred[0]]}"
                    # else:
                    #     self.text2 =  f"neutral"
                    self.prevTime = currentTime 
                if not self.faceMeshOn:
                    # bounding face box
                    cv2.rectangle(image, (bbox_xmin, bbox_ymin), (bbox_xmin + bbox_width, bbox_ymin + bbox_height), (0, 255, 0), 2)
                # get the width and height of the text box
                (textWidth, textHeight) = cv2.getTextSize(self.text2, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)[0]
                # text background highlight
                cv2.rectangle(image, (bbox_xmin-5,bbox_ymin+bbox_height+25-textHeight), (bbox_xmin+textWidth+5,bbox_ymin+bbox_height+35), (70,70,70), cv2.FILLED)
                cv2.putText(image, self.text2, (bbox_xmin,bbox_ymin+bbox_height+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2) 
            else:
                self.text_0 = f"Distance from Camera = Unknown"    
                cv2.putText(image,'No Face Detected', (int((w/2)-200),int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
                cv2.putText(image,'Move Closer to the Centre', (int((w/2)-300),int((h/2)+50)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
                    
            # IPD display
            (textWidthText0, textHeightText0) = cv2.getTextSize(self.text_0, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)[0]
            # text background highlight
            cv2.rectangle(image, (int(w*0.1)-5,int(h*0.95)-textHeightText0-5), (int(w*0.1)+textWidthText0+5,int(h*0.95)+5), (70,70,70), cv2.FILLED)
            cv2.putText(image, self.text_0, (int(w*0.1),int(h*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            
            # FPS display
            fps = round(1/(time.time() - startFrame),2)
            self.fpsFifo.append(fps)
            # self.fpsText = f"FPS = {fps}"
            self.fpsText = f"FPS = {round(np.mean(self.fpsFifo),2)}"
            (textWidthTextFPS, textHeightTextFPS) = cv2.getTextSize(self.fpsText, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)[0]
            # text background highlight
            cv2.rectangle(image, (int(w*0.7)-5,int(h*0.95)-textHeightTextFPS-5), (int(w*0.7)+textWidthTextFPS+5,int(h*0.95)+5), (70,70,70), cv2.FILLED)
            cv2.putText(image, self.fpsText, (int(w*0.7),int(h*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                    
            # convert the frame to RGB format
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # create a QImage from the frame data
            image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)

            # create a QPixmap from the QImage
            pixmap = QPixmap.fromImage(image)

            # scale the QPixmap to fit the label dimensions
            pixmap_scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio)

            # set the QPixmap as the label's pixmap
            self.video_label.setPixmap(pixmap_scaled)
            
            self.frameCount += 1 # increment frame number
        else:
            self.font = QFont('Arial', 24)  # set font and size
            self.font.setWeight(QFont.Bold)  # set font weight to bold for thicker text
            self.video_label.setFont(self.font)
            self.video_label.setText("Camera Not Connected")
            print("Unable to read from camera")

    def closeEvent(self, event):
        '''
        This function releases the video capture and stop the timer when the window is closed
        Parameters:
        None

        Returns:
        Nothing
        '''
        self.cap.release()
        self.timer.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    video_player = MainWindow()
    video_player.show()

    sys.exit(app.exec_())


