from PIL import Image
import numpy as np

import sys
import cv2
from mss import mss
import time
import pyautogui

def grab_drops(sct):
    # drops = {'top': 265, 'left': 750, 'width': 200, 'height': 35}
    drops = {'top': 265, 'left': 1000, 'width': 200, 'height': 35}
    
    img = np.array(sct.grab(drops))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    
    #Display the picture in grayscale
    winname = "Drops"
    cv2.namedWindow(winname) 
    cv2.moveWindow(winname, 830, 1000) 
    cv2.imshow(winname, img)
    # FOR COLLECTING DATA
    cv2.imwrite("d-{}.jpg".format(time.time()), img)
    return img

def grab_wave(sct, character):
    wave = {'top': 125, 'left': 1520, 'width': 130, 'height': 35}
    img = np.array(sct.grab(wave))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    
    #Display the picture in grayscale
    winname = "Wave"
    cv2.namedWindow(winname) 
    cv2.moveWindow(winname, 1280, 1000) 
    cv2.imshow(winname, img)
    # FOR COLLECTING DATA
    cv2.imwrite("{}-{}.jpg".format(character, time.time()), img)
    return img

character = sys.argv[1]
print("Gathering data for {}".format(character))

with mss() as sct:
    # Part of the screen to capture
    while "Screen capturing":
        wave = grab_wave(sct, character)
        # drop = grab_drops(sct)

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break