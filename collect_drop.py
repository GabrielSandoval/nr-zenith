import cv2
import sys
import numpy as np

from mss import mss
from utils import *

print("Gathering data for drops")

drop_boundaries = []
Y_DEVIATION = 3
X_DEVIATION = 5
for r in np.arange(-Y_DEVIATION, Y_DEVIATION):
    for c in np.arange(-X_DEVIATION, X_DEVIATION):
        drop_boundary = {
            'top': BASE_BOUNDARY['top'] + r,
            'left': BASE_BOUNDARY['left'] + c,
            'width': BASE_BOUNDARY['width'],
            'height': BASE_BOUNDARY['height']
        }
        drop_boundaries.append(drop_boundary)
boundary_index = 0

with mss() as sct:
    # Part of the screen to capture
    while "Screen capturing":
        capture_drop(sct, BASE_BOUNDARY)
        grab_drop(sct, drop_boundaries[boundary_index])

        boundary_index += 1
        boundary_index %= len(drop_boundaries)

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break