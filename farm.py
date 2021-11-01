import cv2
import sys
import pyautogui

from mss import mss
from constants import *
from utils import *
from telegram import *

character = sys.argv[1]
print("Farming Lair ({})".format(character))

wave_model  = load_artifact("models\\{}_model.pkl".format(character))
wave_scaler = load_artifact("models\\{}_scaler.pkl".format(character))
drop_model  = load_artifact("models\\drop_model.pkl")
drop_scaler = load_artifact("models\\drop_scaler.pkl")

retry_counter = 0
preparation_counter = 0
wave_counter = 0
drop_counter = 0
no_drop_counter = 0

state = "preparation"

def log(retry_counter, state, counter, limit, message):
    print("[{}][{}][{}%] {}".format(retry_counter, state, round(counter/limit*100, 1), message))
    if (counter == limit - 10 or counter >= limit):# and state == "menu":
        telegram_log("\[{}]\[{}] {}".format(retry_counter, state, message))

with mss() as sct:
    while "Screen capturing":
        if retry_counter > RETRY_LIMIT:
            log(retry_counter, state, retry_counter, RETRY_LIMIT, "RETRY_LIMIT reached. Exiting")
            exit()

        if state == "preparation":
            preparation_counter += 1
            log(retry_counter, state, preparation_counter, PREPARATION_LIMIT, "Preparing for battle")

            if preparation_counter > PREPARATION_LIMIT:
                pyautogui.click(x=930, y=300, interval=2)
                pyautogui.click(x=930, y=900)
                log(retry_counter, state, preparation_counter, PREPARATION_LIMIT, "Battle starts")
                state = "battle"
                preparation_counter = 0
                
        wave = capture_wave(sct, WAVE_BOUNDARY)
        
        if state == "battle":
            height, width = wave.shape
            wave_prediction = wave_model.predict(wave_scaler.transform([wave.reshape(height*width,)]))[0]

            if wave_prediction == 1:
                log(retry_counter, state, wave_counter, WAVE_LIMIT, "Wave 3 Detected")
                wave_counter += 1

            if wave_counter > WAVE_LIMIT:
                log(retry_counter, state, wave_counter, WAVE_LIMIT, "Pressing 'ESC' now.")
                pyautogui.click(x=930, y=300, interval=0.5)
                pyautogui.press('escape')
                state = "menu"
                wave_counter = 0
    
        drop = capture_drop(sct, DROP_BOUNDARY)

        if state == "menu":
            height, width = drop.shape
            drop_prediction = drop_model.predict(drop_scaler.transform([drop.reshape(height*width,)]))[0]

            if drop_prediction == 1:
                log(retry_counter, state, drop_counter, DROP_LIMIT, "PURPLE DROP DETECTED!")
                drop_counter += 1

            if drop_counter > DROP_LIMIT:
                log(retry_counter, state, drop_counter, DROP_LIMIT, "PURPLE DROP FOUND! Exiting.")
                state == "battle"
                no_drop_counter = 0
                drop_counter = 0
                exit()

            if drop_prediction == 0:
                log(retry_counter, state, no_drop_counter, DROP_LIMIT, "No Purple Drop.")
                no_drop_counter += 1

            if no_drop_counter > DROP_LIMIT:
                log(retry_counter, state, no_drop_counter, DROP_LIMIT, "No Purple Drop. Quitting Battle")

                pyautogui.click(x=930, y=775, interval=2)
                pyautogui.click(x=930, y=775)
                retry_counter += 1
                state = "preparation"
                no_drop_counter = 0
                drop_counter = 0

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break