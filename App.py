#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
from PIL import Image
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import sys
import cv2
from mss import mss
import time
import pyautogui

character = sys.argv[1]
print("Farming Lair: {}".format(character))

# In[2]:

def process_directory(directory, character=None):
    features = []
    for filename in os.listdir(directory):
        if character != None and filename.startswith("{}-".format(character)):
            filepath = os.path.join(directory, filename)
            features.append(preprocess(filepath))
            # print(filepath)
        elif character == None and filename.endswith(".jpg"):
            filepath = os.path.join(directory, filename)
            features.append(preprocess(filepath))
    return features

def preprocess(filepath):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    height, width = image.shape
    return image.reshape(height*width,)

# In[3]:


def train_wave(character):
    negative_samples = process_directory("negative-wave", character=character)
    negative_samples = pd.DataFrame(np.array(negative_samples, dtype=np.float32))
    negative_samples["label"] = 0

    positive_samples = process_directory("positive-wave", character=character)
    positive_samples = pd.DataFrame(np.array(positive_samples, dtype=np.float32))
    positive_samples["label"] = 1

    dataset = pd.concat([negative_samples, positive_samples])

    X = dataset.drop('label', axis=1)
    Y = dataset["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    model = MLPClassifier(hidden_layer_sizes=(16, 16, 16, 16), max_iter=1000, alpha=1e-5, solver='adam', verbose=0)
    model.fit(X_train,y_train)
    predictions = model.predict(scaler.transform(X_test))

    f1score = f1_score(y_test, predictions).round(4)
    accuracy = accuracy_score(y_test, predictions).round(4)
    cm = confusion_matrix(y_test,predictions)

    print("F1Score: {}".format(f1score))
    print("Accuracy: {}".format(accuracy))
    print("Confusion Matrix:\n{}".format(cm))
    
    return model, scaler
wave_model, wave_scaler = train_wave(character)


# In[4]:


def train_drop():
    negative_samples = process_directory("negative-drop")
    negative_samples = pd.DataFrame(np.array(negative_samples, dtype=np.float32))
    negative_samples["label"] = 0

    positive_samples = process_directory("positive-drop")
    positive_samples = pd.DataFrame(np.array(positive_samples, dtype=np.float32))
    positive_samples["label"] = 1

    dataset = pd.concat([negative_samples, positive_samples])

    X = dataset.drop('label', axis=1)
    Y = dataset["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    model = MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=1000, alpha=1e-4, solver='adam', verbose=0)
    model.fit(X_train,y_train)
    predictions = model.predict(scaler.transform(X_test))

    f1score = f1_score(y_test, predictions).round(4)
    accuracy = accuracy_score(y_test, predictions).round(4)
    cm = confusion_matrix(y_test,predictions)

    print("F1Score: {}".format(f1score))
    print("Accuracy: {}".format(accuracy))
    print("Confusion Matrix:\n{}".format(cm))
    
    return model, scaler
drop_model, drop_scaler = train_drop()


# In[5]:


def grab_drops(sct):
    # drops = {'top': 265, 'left': 750, 'width': 200, 'height': 35}
    drops = {'top': 265, 'left': 1000, 'width': 200, 'height': 35}
    
    # Get raw pixels from the screen, save it to a Numpy array
    img = np.array(sct.grab(drops))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    
    #Display the picture in grayscale
    winname = "Drops"
    cv2.namedWindow(winname) 
    cv2.moveWindow(winname, 830, 1000) 
    cv2.imshow(winname, img)
    return img
    
def grab_wave(sct):
    wave = {'top': 125, 'left': 1520, 'width': 130, 'height': 35}
    img = np.array(sct.grab(wave))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    
    #Display the picture in grayscale
    winname = "Wave"
    cv2.namedWindow(winname) 
    cv2.moveWindow(winname, 1280, 1000) 
    cv2.imshow(winname, img)
    return img


# In[7]:

PREPARATION_LIMIT = 500
WAVE_LIMIT = 100
DROP_LIMIT = 50
MASTER_COUNTER_LIMIT = 50

master_counter = 0
preparation_counter = 0
wave_counter = 0
drop_counter = 0
no_drop_counter = 0

state = "preparation"

with mss() as sct:
    # Part of the screen to capture

    while "Screen capturing":
        last_time = time.time()
        
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
                
        wave = grab_wave(sct)
        
        if state == "battle":
            height, width = wave.shape
            wave_prediction = wave_model.predict(wave_scaler.transform([wave.reshape(height*width,)]))[0]

            if wave_prediction == 1:
                print("[{}][{}][{}%] WAVE 3 Detected".format(master_counter, state, round(wave_counter/WAVE_LIMIT*100, 1)))
                wave_counter += 1

            if wave_counter > WAVE_LIMIT:
                print("[{}][{}][{}%] WAVE 3 Detected > {} times. Pressing 'ESC' now.".format(master_counter, state, round(wave_counter/WAVE_LIMIT*100, 1), WAVE_LIMIT))
                pyautogui.press('escape')
                state = "menu"
                wave_counter = 0
        
        drop = grab_drops(sct)

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

#         print("fps: {}".format(1 / (time.time() - last_time)))

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
