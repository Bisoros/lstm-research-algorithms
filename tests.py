from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import os
import numpy as np
import csv

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import classification_report

# split a univariate sequence


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


data = []
n_steps = 6
train_split = 0.25

for file in os.listdir('./data'):
    filename = os.fsdecode(file)
    arr = np.genfromtxt('./data/' + filename, delimiter=',')
    if len(arr) > n_steps:
        data.append(arr)


sep = int(len(data) * train_split)

Xf = []
yf = []

for instance in data[:sep]:
    X, y = split_sequence(instance, n_steps)
    Xf.append(X)
    yf.append(y)

Xf = np.vstack(Xf)
yf = np.hstack(yf)

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


for model in classifiers:
    model.fit(X, y)

    pred = []
    truth = []
    for instance in data[sep:]:
        X, y = split_sequence(instance, n_steps)
        pred.append(model.predict(X))
        truth.append(y)

    pred = np.hstack(pred)
    truth = np.concatenate(truth).reshape(-1, 1)

    print(type(model).__name__)
    print(classification_report(truth, pred))
