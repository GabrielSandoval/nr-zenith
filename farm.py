import cv2
import sys
import pyautogui

from mss import mss
from constants import *
from utils import *

character = sys.argv[1]
print("Farming Lair ({})".format(character))

wave_model, wave_scaler = train_wave(character)
drop_model, drop_scaler = train_drop()
master_counter = 0
preparation_counter = 0
wave_counter = 0
drop_counter = 0
no_drop_counter = 0

state = "preparation"

with mss() as sct:
    while "Screen capturing":
        if master_counter >= MASTER_COUNTER_LIMIT:
            print("MASTER_COUNTER_LIMIT reached. Exiting.")
            exit()

        if state == "preparation":
            print("[{}][{}][{}%] Preparing for battle".format(master_counter, state, round(preparation_counter/PREPARATION_LIMIT*100, 1)))
            preparation_counter += 1
            
            if preparation_counter > PREPARATION_LIMIT:
                pyautogui.click(x=930, y=300, interval=3)
                pyautogui.click(x=930, y=900)
                state = "battle"
                preparation_counter = 0
                
        wave = capture_wave(sct, WAVE_BOUNDARY)
        
        if state == "battle":
            height, width = wave.shape
            wave_prediction = wave_model.predict(wave_scaler.transform([wave.reshape(height*width,)]))[0]

            if wave_prediction == 1:
                print("[{}][{}][{}%] WAVE 3 Detected".format(master_counter, state, round(wave_counter/WAVE_LIMIT*100, 1)))
                wave_counter += 1

            if wave_counter > WAVE_LIMIT:
                print("[{}][{}][{}%] WAVE 3 Detected > {} times. Pressing 'ESC' now.".format(master_counter, state, round(wave_counter/WAVE_LIMIT*100, 1), WAVE_LIMIT))
                pyautogui.click(x=930, y=775)
                pyautogui.press('escape')
                state = "menu"
                wave_counter = 0
        
        drop = capture_drop(sct, DROP_BOUNDARY)

        if state == "menu":
            height, width = drop.shape
            drop_prediction = drop_model.predict(drop_scaler.transform([drop.reshape(height*width,)]))[0]

            if drop_prediction == 1:
                print("[{}][{}][{}%] PURPLE DROP Detected.".format(master_counter, state, round(drop_counter/DROP_LIMIT*100, 1)))
                drop_counter += 1

            if drop_counter > DROP_LIMIT:
                print("[{}][{}][{}%] PURPLE DROP Detected > {} times.".format(master_counter, state, round(drop_counter/DROP_LIMIT*100, 1), DROP_LIMIT))
                state == "battle"
                drop_counter = 0
                print("Found purple after {} tries".format(master_counter))
                exit()

            if drop_prediction == 0:
                print("[{}][{}][{}%] NO PURPLE DROP".format(master_counter, state, round(no_drop_counter/DROP_LIMIT*100, 1)))
                no_drop_counter += 1

            if no_drop_counter > DROP_LIMIT:
                print("[{}][{}][{}%] NO PURPLE DROP. Quitting battle.".format(master_counter, state, round(no_drop_counter/DROP_LIMIT*100, 1)))
                pyautogui.click(x=930, y=775, interval=2)
                pyautogui.click(x=930, y=775)
                master_counter += 1
                state = "preparation"
                no_drop_counter = 0
                drop_counter = 0

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break