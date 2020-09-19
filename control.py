import os
import numpy as np
import csv

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
    print(file)
    filename = os.fsdecode(file)
    arr = np.genfromtxt('./data/' + filename, delimiter=',')
    if len(arr) > n_steps:
        data.append(arr)
    else:
        print(arr)

print('Nr. samles: ', len(data))
aux = [len(i) for i in data]
from statistics import mean, median, mode
print(mean(aux))
print(median(aux))
print(mode(aux))
print(min(aux))
print(max(aux))


sep = int(len(data) * train_split)

truth = []
greedy1 = []
greedy2 = []
for instance in data[sep:]:
    X, y = split_sequence(instance, n_steps)
    for arr in X:
        greedy1.append(arr[-1])
        greedy2.append(1 - arr[-1])
    X = X.reshape((X.shape[0], X.shape[1], 1))
    truth.append(y)

truth = np.concatenate(truth).reshape(-1, 1)
zeros = np.zeros(truth.shape)
ones = np.ones(truth.shape)
rand = np.random.choice([0, 1], size=truth.shape)
greedy1 = np.array(greedy1)
greedy2 = np.array(greedy2)

print(classification_report(truth, zeros))
print(classification_report(truth, ones))
print(classification_report(truth, rand))
print(classification_report(truth, greedy1))
print(classification_report(truth, greedy2))
