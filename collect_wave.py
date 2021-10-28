from PIL import Image
import numpy as np

import sys
import cv2
from mss import mss
import time

def grab_wave(sct, character, wave_boundary):
    img = np.array(sct.grab(wave_boundary))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    # FOR COLLECTING DATA
    cv2.imwrite("akeha\\{}-{}.jpg".format(character, time.time()), img)

def show_wave(sct, character, wave_boundary):
    img = np.array(sct.grab(wave_boundary))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    
    #Display the picture in grayscale
    winname = "Wave"
    cv2.namedWindow(winname) 
    cv2.moveWindow(winname, 1280, 1000) 
    cv2.imshow(winname, img)
    return img

character = sys.argv[1]
print("Gathering data for {}".format(character))

BASE_BOUNDARY = {'top': 130, 'left': 1600, 'width': 50, 'height': 25}
wave_boundaries = []
Y_DEVIATION = 3
X_DEVIATION = 5

for r in np.arange(-Y_DEVIATION, Y_DEVIATION):
    for c in np.arange(-X_DEVIATION, X_DEVIATION):
        wave_boundary = {'top': BASE_BOUNDARY['top'] + r, 'left': BASE_BOUNDARY['left'] + c, 'width': 50, 'height': 25}
        wave_boundaries.append(wave_boundary)
        
boundary_index = 0

with mss() as sct:
    # Part of the screen to capture
    while "Screen capturing":
        show_wave(sct, character, BASE_BOUNDARY)
        grab_wave(sct, character, wave_boundaries[boundary_index])

        boundary_index += 1
        boundary_index %= len(wave_boundaries)

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break