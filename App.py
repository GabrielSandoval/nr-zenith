#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
from PIL import Image
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def process_directory(directory, label):
    features = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"): 
            filepath = os.path.join(directory, filename)
#             print(filepath)
            features.append(preprocess(filepath))
    return features

def preprocess(filepath):
    image = Image.open(filepath).convert('L')
    image_array = np.array(image)
    height, width = image_array.shape
    return list(image_array.reshape(height*width,))


# In[2]:


def train_wave():
    negative_samples = process_directory("negative-wave", 0)
    negative_samples = pd.DataFrame(np.array(negative_samples, dtype=np.float32)/255)
    negative_samples["label"] = 0

    positive_samples = process_directory("positive-wave", 1)
    positive_samples = pd.DataFrame(np.array(positive_samples, dtype=np.float32)/255)
    positive_samples["label"] = 1

    dataset = pd.concat([negative_samples, positive_samples])

    X = dataset.drop('label', axis=1)
    Y = dataset["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    model = MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=500, alpha=1e-4, solver='adam', verbose=0)
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)

    f1score = f1_score(y_test, predictions).round(4)
    accuracy = accuracy_score(y_test, predictions).round(4)
    cm = confusion_matrix(y_test,predictions)

    print("F1Score: {}".format(f1score))
    print("Accuracy: {}".format(accuracy))
    print("Confusion Matrix:\n{}".format(cm))
    
    return model
wave_model = train_wave()


# In[3]:


def train_drop():
    negative_samples = process_directory("negative-drop", 0)
    negative_samples = pd.DataFrame(np.array(negative_samples, dtype=np.float32)/255)
    negative_samples["label"] = 0

    positive_samples = process_directory("positive-drop", 1)
    positive_samples = pd.DataFrame(np.array(positive_samples, dtype=np.float32)/255)
    positive_samples["label"] = 1

    dataset = pd.concat([negative_samples, positive_samples])

    X = dataset.drop('label', axis=1)
    Y = dataset["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    
    model = MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=1000, alpha=1e-4, solver='adam', verbose=0)
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)

    f1score = f1_score(y_test, predictions).round(4)
    accuracy = accuracy_score(y_test, predictions).round(4)
    cm = confusion_matrix(y_test,predictions)

    print("F1Score: {}".format(f1score))
    print("Accuracy: {}".format(accuracy))
    print("Confusion Matrix:\n{}".format(cm))
    
    return model
drop_model = train_drop()


# In[4]:


import cv2
from mss import mss
import time
import pyautogui

import ctypes
user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
screensize


# In[5]:


def grab_drops(sct):
#     monitor = {'top': 200, 'left': 2400, 'width': 300, 'height': 150}
    monitor = {'top': 200, 'left': 2700, 'width': 300, 'height': 150}
    
    # Get raw pixels from the screen, save it to a Numpy array
    img = np.array(sct.grab(monitor))
    
    # Display the picture
    # cv2.imshow("OpenCV/Numpy normal", img)

    #Display the picture in grayscale
    
    winname = "Drops"
    cv2.namedWindow(winname) 
    cv2.moveWindow(winname, 450, 10) 
    cv2.imshow(winname, cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))
    return img
    
def grab_wave(sct):
    wave = {'top': 100, 'left': 3200, 'width': 200, 'height': 100}
    img = np.array(sct.grab(wave))
    
    
    winname = "Wave"
    cv2.namedWindow(winname) 
    cv2.moveWindow(winname, 10, 10) 
    cv2.imshow(winname, cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))
    return img


# In[ ]:


WAVE_LIMIT = 100
DROP_LIMIT = 50
state = "battle"

with mss() as sct:
    # Part of the screen to capture

    wave_counter = 0
    drop_counter = 0
    no_drop_counter = 0
    
    while "Screen capturing":
        last_time = time.time()
        
        wave = grab_wave(sct)
        wave = cv2.cvtColor(wave, cv2.COLOR_BGRA2GRAY)/255
        height, width = wave.shape
        wave_prediction = wave_model.predict([wave.reshape(height*width,)])[0]

        if state == "battle" and wave_prediction == 1:
            print("[{}][{}/{}] WAVE 3 Detected".format(state, wave_counter, WAVE_LIMIT))
            wave_counter += 1

            if wave_counter > WAVE_LIMIT:
                print("[{}][{}/{}] WAVE 3 Detected > {} times. Pressing 'ESC' now.".format(state, wave_counter, WAVE_LIMIT, WAVE_LIMIT))
                pyautogui.press('escape')
                state = "menu"
                wave_counter = 0
        
        drop = grab_drops(sct)
        drop = cv2.cvtColor(drop, cv2.COLOR_BGRA2GRAY)/255
        height, width = drop.shape

        drop_prediction = drop_model.predict([drop.reshape(height*width,)])[0]

        if state == "menu":
            if drop_prediction == 1:
                print("[{}][{}/{}] PURPLE DROP Detected.".format(state, drop_counter, DROP_LIMIT))
                drop_counter += 1

                if drop_counter > DROP_LIMIT:
                    print("[{}][{}/{}] PURPLE DROP Detected > {} times.".format(state, drop_counter, DROP_LIMIT, DROP_LIMIT))
                    pyautogui.click(x=2500, y=875)
                    state == "battle"
                    drop_counter = 0

            if drop_prediction == 0:
                print("[{}][{}/{}] NO PURPLE DROP".format(state, no_drop_counter, DROP_LIMIT))
                no_drop_counter += 1

                if no_drop_counter > DROP_LIMIT:
                    print("[{}][{}/{}] NO PURPLE DROP. Quitting battle.".format(state, no_drop_counter, DROP_LIMIT))
                    pyautogui.click(x=2650, y=775, interval=2.5)
                    pyautogui.click(x=2665, y=775, interval=0.25)
                    state = "battle"
                    no_drop_counter = 0

#         print("fps: {}".format(1 / (time.time() - last_time)))

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

