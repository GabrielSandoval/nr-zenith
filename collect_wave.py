import cv2
import numpy as np
import sys

from mss import mss
from utils import *

character = sys.argv[1]
print("Gathering data for {}".format(character))

wave_boundaries = []
Y_DEVIATION = 3
X_DEVIATION = 5

for r in np.arange(-Y_DEVIATION, Y_DEVIATION):
    for c in np.arange(-X_DEVIATION, X_DEVIATION):
        wave_boundary = {
            'top': BASE_BOUNDARY['top'] + r,
            'left': BASE_BOUNDARY['left'] + c,
            'width': BASE_BOUNDARY['width'],
            'height': BASE_BOUNDARY['height']
        }
        wave_boundaries.append(wave_boundary)
        
boundary_index = 0

with mss() as sct:
    # Part of the screen to capture
    while "Screen capturing":
        capture_wave(sct, BASE_BOUNDARY)
        grab_wave(sct, character, wave_boundaries[boundary_index])

        boundary_index += 1
        boundary_index %= len(wave_boundaries)

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break