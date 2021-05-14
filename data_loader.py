# !/usr/bin/env python3

import numpy as np
import os 
import sys
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import tensorflow as tf


from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.layers import *
from fourier import eemd, ceemdan
from config import *


class DataLoader():
    def __init__(self, path = OUTPUT_DIR, classes = N_CLASSES, batch_size = BATCH_SIZE):
        self.path = path
        self.classes = classes
        self.batch_size = batch_size
    
    def load_single_npz(self, file):
        with np.load(file) as f:
            data = f["x"], labels = f["y"]
        return data, labels

    def list_npz_files(self):
        files = [self.path+f for f in listdir(self.path) if isfile(join(self.path, f))]
        return files
    def get_data(self):
        x_train, y_train = [], []
        for k in self.list_npz_files():
            with np.load(k) as f:
                data = f["x"]
                labels = f["y"]
                labels = np.expand_dims(labels,axis=1)
                for i in range(len(data)):
                    a = np.zeros((5,))
                    a[labels[i]] = 1 
                    a = np.reshape(a, (None,5))
                    x_train.append(eemd(data[i]))
                    y_train.append(a)

        return np.asarray(x_train), np.asarray(y_train)
    def load_batch(self):
        x,y = self.get_data()
        shuffle = np.random.permutation(len(x))
        start = 0
        x = x[shuffle]
        y = y[shuffle]
        while start + self.batch_size <= len(x):
            yield x[start:start+self.batch_size], y[start:start+self.batch_size]
            start += self.batch_size
    def construct_dataset(self):
        gen = self.load_batch()
        dataset = tf.data.Dataset.from_generator(
                lambda: gen,
                output_signature=(
                tf.TensorSpec(shape=(BATCH_SIZE, 3000,1), dtype=tf.float32),
                tf.TensorSpec(shape=(BATCH_SIZE, 5,1), dtype=tf.int32)))## problem with pooling

        return dataset

        
def list_npz_files(filepath=OUTPUT_DIR):
    files = [filepath+f for f in listdir(filepath) if isfile(join(filepath, f))]
    return files

def get_data(filepath=OUTPUT_DIR):
    x_train, y_train = [], []
    for k in tqdm(list_npz_files(filepath= filepath)):
        with np.load(k) as f:
            data = f["x"]
            labels = f["y"]
            #labels = np.expand_dims(labels,axis=1)
            for i in range(len(data)):
                a = np.zeros((5))
                a[labels[i]] = 1 
                a = np.reshape(a, (1,5))
                x_train.append(data[i])
                y_train.append(a)

    return np.asarray(x_train), np.asarray(y_train)


def generator(filepath=OUTPUT_DIR):
    for k in list_npz_files(filepath = filepath)[:5]:
        try:
            with np.load(k) as f:
                data = f["x"]
                labels = f["y"]
                labels = np.expand_dims(labels,axis=1)
                for i in range(len(data)):
                    a = np.zeros((5))
                    a[labels[it]] = 1 
                    #a = np.reshape(a, (1,5))
                    yield data[i],a
        except StopIteration: break
        except: pass

def batch_data(x, y, batch_size):
    shuffle = np.random.permutation(len(x))
    start = 0
#     from IPython.core.debugger import Tracer; Tracer()()
    x = x[shuffle]
    y = y[shuffle]
    while start + batch_size <= len(x):
        yield x[start:start+batch_size], y[start:start+batch_size]
        start += batch_size


def conv_model(input_shape = INPUT_SHAPE):
    act = tf.nn.leaky_relu
    model = Sequential([
        Conv1D(64, 1, activation=act, input_shape = (3000,1)),
        BatchNormalization(),
        LSTM(64),
        Dense(5, activation='softmax')
    ])
    model.compile(optimizer='adam', loss = 'categorical_crossentropy')
    model.build((3000,1))
    model.summary()
    return model

def construct_dataset(repeat = 5):
    data, labels = get_data()
    gen = batch_data(data, labels, BATCH_SIZE)
    dataset = tf.data.Dataset.from_generator(
            lambda: gen,
            output_signature=(
            tf.TensorSpec(shape=(BATCH_SIZE, 3000,1), dtype=tf.float32),
            tf.TensorSpec(shape=(BATCH_SIZE, 5,1), dtype=tf.int32)))## problem with pooling

    return dataset

def split_dataset(dataset):
    DATASET_SIZE = tf.data.experimental.cardinality(dataset).numpy()
    train_size = int(0.85 * DATASET_SIZE)
    val_size = int(0.15 * DATASET_SIZE)
    test_size = int(0.05 * DATASET_SIZE)

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)
    return train_dataset, val_dataset, test_dataset
