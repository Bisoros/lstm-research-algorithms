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


# configuration
n_steps = 6
train_split = 0.25
epochs = 100

# read data
data = []
for file in os.listdir('./data'):
    filename = os.fsdecode(file)
    arr = np.genfromtxt('./data/' + filename, delimiter=',')
    if len(arr) > n_steps:
        data.append(arr)

# construct network
model = Sequential()
model.add(LSTM(50, activation='relu',
               return_sequences=True, input_shape=(n_steps, 1)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# train
sep = int(len(data) * train_split)

for i in range(epochs):
    for instance in data[:sep]:
        X, y = split_sequence(instance, n_steps)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        model.fit(X, y, epochs=1, verbose=1)

# test
pred = []
truth = []
for instance in data[sep:]:
    X, y = split_sequence(instance, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    pred.append(model.predict(X, verbose=0))
    truth.append(y)

pred = np.round(np.vstack(pred))
truth = np.concatenate(truth).reshape(-1, 1)

print(classification_report(truth, pred))
