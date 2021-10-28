import os
import time
import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def process_directory(directory, character="all"):
    features = []
    for filename in os.listdir(directory):
        if character == "all" and filename.endswith(".jpg"):
            filepath = os.path.join(directory, filename)
            features.append(preprocess(filepath))
        elif character != None and filename.startswith("{}-".format(character)):
            filepath = os.path.join(directory, filename)
            features.append(preprocess(filepath))
    return features

def preprocess(filepath):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    height, width = image.shape
    return image.reshape(height*width,)

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

def capture_wave(sct, wave_boundary):
    img = np.array(sct.grab(wave_boundary))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    
    #Display the picture in grayscale
    winname = "Wave"
    cv2.namedWindow(winname) 
    cv2.moveWindow(winname, 1280, 1000) 
    cv2.imshow(winname, img)
    return img

def capture_drop(sct, drop_boundary):
    img = np.array(sct.grab(drop_boundary))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    
    #Display the picture in grayscale
    winname = "Drop"
    cv2.namedWindow(winname) 
    cv2.moveWindow(winname, 850, 1000) 
    cv2.imshow(winname, img)
    return img

def grab_wave(sct, character, wave_boundary):
    img = np.array(sct.grab(wave_boundary))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    # FOR COLLECTING DATA
    cv2.imwrite("{}\\{}-{}.jpg".format(character, character, time.time()), img)

def grab_drop(sct, drop_boundary):
    img = np.array(sct.grab(drop_boundary))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    # FOR COLLECTING DATA
    cv2.imwrite("d-{}.jpg".format(time.time()), img)