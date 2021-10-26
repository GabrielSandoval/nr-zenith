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

import cv2
from mss import mss
import time
import pyautogui

import ctypes
user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
screensize


# In[2]:


def process_directory(directory, label):
    features = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"): 
            filepath = os.path.join(directory, filename)
#             print(filepath)
            features.append(preprocess(filepath))
    return features

def preprocess(filepath):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    height, width = image.shape
    return image.reshape(height*width,)


# In[3]:


def train_wave():
    negative_samples = process_directory("negative-wave", 0)
    negative_samples = pd.DataFrame(np.array(negative_samples, dtype=np.float32))
    negative_samples["label"] = 0

    positive_samples = process_directory("positive-wave", 1)
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
wave_model, wave_scaler = train_wave()


# In[4]:


def train_drop():
    negative_samples = process_directory("negative-drop", 0)
    negative_samples = pd.DataFrame(np.array(negative_samples, dtype=np.float32))
    negative_samples["label"] = 0

    positive_samples = process_directory("positive-drop", 1)
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
#     monitor = {'top': 265, 'left': 750, 'width': 200, 'height': 35}
    drops = {'top': 265, 'left': 1000, 'width': 200, 'height': 35}
    
    # Get raw pixels from the screen, save it to a Numpy array
    img = np.array(sct.grab(drops))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    
    #Display the picture in grayscale
    winname = "Drops"
    cv2.namedWindow(winname) 
    cv2.moveWindow(winname, 830, 1000) 
    cv2.imshow(winname, img)
    # FOR COLLECTING DATA
    # cv2.imwrite("d-{}.png".format(time.time()), img)
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
    # FOR COLLECTING DATA
#     cv2.imwrite("w-{}.png".format(time.time()), img)
    return img


# In[6]:


# # FOR COLLECTING DATA

# PREPARATION_LIMIT = 400
# WAVE_LIMIT = 100
# DROP_LIMIT = 50

# preparation_counter = 0
# wave_counter = 0
# drop_counter = 0
# no_drop_counter = 0

# state = "preparation"

# with mss() as sct:
#     # Part of the screen to capture
#     while "Screen capturing":
#         wave = grab_wave(sct)
#         drop = grab_drops(sct)

#         # Press "q" to quit
#         if cv2.waitKey(25) & 0xFF == ord("q"):
#             cv2.destroyAllWindows()
#             break


# In[7]:


PREPARATION_LIMIT = 400
WAVE_LIMIT = 100
DROP_LIMIT = 50

preparation_counter = 0
wave_counter = 0
drop_counter = 0
no_drop_counter = 0

state = "preparation"

with mss() as sct:
    # Part of the screen to capture

    while "Screen capturing":
        last_time = time.time()
        
        if state == "preparation":
            print("[{}][{}/{}] Preparing for battle".format(state, preparation_counter, PREPARATION_LIMIT))
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
                print("[{}][{}/{}] WAVE 3 Detected".format(state, wave_counter, WAVE_LIMIT))
                wave_counter += 1

            if wave_counter > WAVE_LIMIT:
                print("[{}][{}/{}] WAVE 3 Detected > {} times. Pressing 'ESC' now.".format(state, wave_counter, WAVE_LIMIT, WAVE_LIMIT))
                pyautogui.press('escape')
                state = "menu"
                wave_counter = 0
        
        drop = grab_drops(sct)

        if state == "menu":
            height, width = drop.shape
            drop_prediction = drop_model.predict(drop_scaler.transform([drop.reshape(height*width,)]))[0]

            if drop_prediction == 1:
                print("[{}][{}/{}] PURPLE DROP Detected.".format(state, drop_counter, DROP_LIMIT))
                drop_counter += 1

            if drop_counter > DROP_LIMIT:
                print("[{}][{}/{}] PURPLE DROP Detected > {} times.".format(state, drop_counter, DROP_LIMIT, DROP_LIMIT))
                state == "battle"
                drop_counter = 0
                exit()

            if drop_prediction == 0:
                print("[{}][{}/{}] NO PURPLE DROP".format(state, no_drop_counter, DROP_LIMIT))
                no_drop_counter += 1

            if no_drop_counter > DROP_LIMIT:
                print("[{}][{}/{}] NO PURPLE DROP. Quitting battle.".format(state, no_drop_counter, DROP_LIMIT))
                pyautogui.click(x=930, y=775, interval=2)
                pyautogui.click(x=930, y=775)
                state = "preparation"
                no_drop_counter = 0

#         print("fps: {}".format(1 / (time.time() - last_time)))

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

