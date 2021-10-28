import time
from mss import mss

from utils import *
from constants import *

character = "all"
print("Testing detectors: {}".format(character))
wave_model, wave_scaler = train_wave(character)
drop_model, drop_scaler = train_drop()

with mss() as sct:
    while "Screen capturing":
        last_time = time.time()

        wave = capture_wave(sct, WAVE_BOUNDARY)
        height, width = wave.shape
        wave_prediction = wave_model.predict(wave_scaler.transform([wave.reshape(height*width,)]))[0]
        
        drop = capture_drop(sct, DROP_BOUNDARY)
        height, width = drop.shape
        drop_prediction = drop_model.predict(drop_scaler.transform([drop.reshape(height*width,)]))[0]

        fps = 1 / (time.time() - last_time)
        print("WAVE 3 DETECTED: {} ; DROP DETECTED: {} ; FPS: {}".format(wave_prediction, drop_prediction, fps))

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break