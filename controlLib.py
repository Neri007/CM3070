"""
File: controlLib.py

Class CreateButton

Main application window for Face Emotion Recognition

Author: Nerissa Goh

Version:
0.1     20 Jul 23   new today
1.0     21 Jul 23   released as prelim product

"""
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QGraphicsDropShadowEffect

class CreateButton(QPushButton):
    def __init__(self,text): # class for buttons
        super().__init__(text)
        self.setStyleSheet(
            "background-color: #dbab30;"
            "border: 5px solid #dbab30;"
            "border-radius: 20px;"
            "color: black;"
            "padding: 10px 20px;"
        )
        font = QFont('Arial', 24)  # Another example with a different font family and fontsize
        self.setFont(font)
        
        # Create a QGraphicsDropShadowEffect for the button
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)  # blur radius to control the shadow's blur level
        shadow.setXOffset(5)      # X offset to control the shadow's horizontal position
        shadow.setYOffset(5)      # Y offset to control the shadow's vertical position

        # Apply the shadow effect to the button
        self.setGraphicsEffect(shadow)

def updateTemplateImages(imageTemplateName, imageTemplateNum, blackImage, leftImage_label, rightImage_label):
    '''
    This function updates the left and right template images based on imageTemplateNum
    
    Parameters:
    imageTemplateName (list od str): template image filenames
    imageTemplateNum (int): current template image number
    blackImage (image, np array(393,290,3),uint8): black image
    leftImage_label (QLabel Obj): QLabel for left image template
    rightImage_label (QLabel Obj): QLabel for right image template
    
    Returns:
    Nothing
    '''
    # left template image
    try:
        imageLeft = cv2.cvtColor(cv2.resize(cv2.imread(f"./resources/images/{imageTemplateName[imageTemplateNum][0]}.jpg", cv2.IMREAD_COLOR), (290,393)), cv2.COLOR_BGR2RGB)
    except Exception as e:
        imageLeft = blackImage
        print(f"./resources/images/{imageTemplateName[imageTemplateNum][0]}.jpg does not exist",e)
        
    # convert to QImage format
    height, width, channels = imageLeft.shape
    bytes_per_line = channels * width
    imageLeftQimage = QImage(imageLeft.data, width, height, bytes_per_line, QImage.Format_RGB888)
    # convert to Pixmap
    leftImage_pixel = QPixmap.fromImage(imageLeftQimage)
    
    # set to label
    leftImage_label.setPixmap(leftImage_pixel)
    
    # right template image
    try:
        imageRight = cv2.cvtColor(cv2.resize(cv2.imread(f"./resources/images/{imageTemplateName[imageTemplateNum][1]}.jpg", cv2.COLOR_BGR2RGB), (290,393)), cv2.COLOR_BGR2RGB)
    except Exception as e:
        imageRight = blackImage
        print(f"./resources/images/{imageTemplateName[imageTemplateNum][1]}.jpg does not exist",e)
        
    # convert to QImage format
    height, width, channels = imageRight.shape
    bytes_per_line = channels * width
    imageRightQimage = QImage(imageRight.data, width, height, bytes_per_line, QImage.Format_RGB888)
    # convert to Pixmap
    rightImage_pixel = QPixmap.fromImage(imageRightQimage)
    
    # set to label
    rightImage_label.setPixmap(rightImage_pixel)

def next_tab(tabNum, stop_button, right_button, left_button, start_button, mesh_button):
    '''
    This function move tab selection to the next
    Parameters:
    tabNum (int): current tab position
    stop_button (button obj): stop video button
    right_button (button obj): next image button
    left_button (button obj): previous image button
    start_button (button obj): start video button
    mesh_button (button obj): toggle face mesh button
    
    Returns:
    Nothing
    '''
        
    # update button border select
    if tabNum == 0:
        stop_button.setStyleSheet(
            "background-color: #dbab30;"
            "border: 5px solid #dbab30;"
            "border-radius: 20px;"
            "color: black;"
            "padding: 10px 20px;"
        )
        right_button.setStyleSheet(
            "background-color: #dbab30;"
            "border: 5px solid red;"
            "border-radius: 20px;"
            "color: black;"
            "padding: 10px 20px;"
        )
    elif tabNum == 1:
        right_button.setStyleSheet(
            "background-color: #dbab30;"
            "border: 5px solid #dbab30;"
            "border-radius: 20px;"
            "color: black;"
            "padding: 10px 20px;"
        )
        left_button.setStyleSheet(
            "background-color: #dbab30;"
            "border: 5px solid red;"
            "border-radius: 20px;"
            "color: black;"
            "padding: 10px 20px;"
        )
    elif tabNum == 2:
        left_button.setStyleSheet(
            "background-color: #dbab30;"
            "border: 5px solid #dbab30;"
            "border-radius: 20px;"
            "color: black;"
            "padding: 10px 20px;"
        )
        start_button.setStyleSheet(
            "background-color: #dbab30;"
            "border: 5px solid red;"
            "border-radius: 20px;"
            "color: black;"
            "padding: 10px 20px;"
        )
    elif tabNum == 3:
        start_button.setStyleSheet(
            "background-color: #dbab30;"
            "border: 5px solid #dbab30;"
            "border-radius: 20px;"
            "color: black;"
            "padding: 10px 20px;"
        )
        mesh_button.setStyleSheet(
            "background-color: #dbab30;"
            "border: 5px solid red;"
            "border-radius: 20px;"
            "color: black;"
            "padding: 10px 20px;"
        )
    elif tabNum == 4:
        mesh_button.setStyleSheet(
            "background-color: #dbab30;"
            "border: 5px solid #dbab30;"
            "border-radius: 20px;"
            "color: black;"
            "padding: 10px 20px;"
        )
        stop_button.setStyleSheet(
            "background-color: #dbab30;"
            "border: 5px solid red;"
            "border-radius: 20px;"
            "color: black;"
            "padding: 10px 20px;"
        )