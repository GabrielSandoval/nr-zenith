from utils import *

import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

def train_wave(character):
    negative_samples = load_images("data\\negative-wave", character=character)
    negative_samples = pd.DataFrame(np.array(negative_samples, dtype=np.float32))
    negative_samples["label"] = 0

    positive_samples = load_images("data\\positive-wave", character=character)
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
    
    dump_artifact(model, "models/{}_model.pkl".format(character))
    dump_artifact(scaler, "models/{}_scaler.pkl".format(character))

character = sys.argv[1]
print("Training model: {}".format(character))
train_wave(character)