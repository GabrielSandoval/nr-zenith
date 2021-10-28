import os
import time
import cv2
import numpy as np
import pickle

def dump_artifact(artifact, artifact_path):
    with open(artifact_path, 'wb') as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_artifact(artifact_path):
    with open(artifact_path, 'rb') as handle:
        return pickle.load(handle)

def preprocess(filepath):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    height, width = image.shape
    return image.reshape(height*width,)

def load_images(directory, character="all"):
    features = []
    for filename in os.listdir(directory):
        if character == "all" and filename.endswith(".jpg"):
            filepath = os.path.join(directory, filename)
            features.append(preprocess(filepath))
        elif character != None and filename.startswith("{}-".format(character)):
            filepath = os.path.join(directory, filename)
            features.append(preprocess(filepath))
    return features

def capture_wave(sct, wave_boundary):
    img = np.array(sct.grab(wave_boundary))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    
    winname = "Wave"
    cv2.namedWindow(winname) 
    cv2.moveWindow(winname, 1280, 1000) 
    cv2.imshow(winname, img)
    return img

def capture_drop(sct, drop_boundary):
    img = np.array(sct.grab(drop_boundary))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    
    winname = "Drop"
    cv2.namedWindow(winname) 
    cv2.moveWindow(winname, 850, 1000) 
    cv2.imshow(winname, img)
    return img

# FOR COLLECTING DATA
def grab_wave(sct, character, wave_boundary):
    img = np.array(sct.grab(wave_boundary))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    cv2.imwrite("{}\\{}-{}.jpg".format(character, character, time.time()), img)

# FOR COLLECTING DATA
def grab_drop(sct, drop_boundary):
    img = np.array(sct.grab(drop_boundary))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    cv2.imwrite("d-{}.jpg".format(time.time()), img)