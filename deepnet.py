#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import shutil

from os import listdir
from os.path import isfile, join
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.layers import *


from config import *
from data_loader import DataLoader

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=BASE_PATH,
                        help="File path to the CSV or NPY file that contains walking data.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Directory where to save outputs.")
    parser.add_argument("--select_ch", type=str, default=SELECT_CH,
                        help="File path to the trained model used to estimate walking speeds.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    return args

def TwinBlock():
    block = Sequential([
        Conv1D(16, 2, activation = 'relu', padding = 'same'),
        MaxPool1D(2),
        BatchNormalization(),
        Bidirectional(LSTM(32, return_sequences = True), merge_mode = 'concat'),
        Conv1D(32, 2, activation = 'relu', padding = 'same'),
        MaxPool1D(2),
        BatchNormalization(),
        Bidirectional(LSTM(32, return_sequences = True), merge_mode = 'concat'),
    ])
   
    return block

def DenseBlock():
    block = Sequential([
        Concatenate(),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(5, activation='softmax')
    ])
    
    return block

def list_npz_files(filepath=OUTPUT_DIR):
    files = [filepath+f for f in listdir(filepath) if isfile(join(filepath, f))]
    return files


class DeepNetwork(Model):
    def __init__(self):
        super(DeepNetwork, self).__init__()
        # self.input_shape = input_shape
        # self.input_layer = Input(shape = input_shape)
        self.twin_block1 = TwinBlock()
        self.twin_block2 = TwinBlock()
        self.denseBlock = DenseBlock()
    def call(self, inputs):
        #inputs = self.input_layer
        x = self.twin_block1(inputs)
        y = self.twin_block2(inputs)
        z = self.denseBlock([x,y])
        return z
    def plot_history(self, history):
        ## ACCURACY ## 
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        ## LOSS ## 
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

def initDeepNet():
    deep = DeepNetwork()
    deep.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    deep.build((None, 3000,1))
    deep.summary()
    return deep

def train(model, dataset,  args = sys.argv[1:]):
    er = EarlyStopping(monitor = 'accuracy', patience = 2)
    reduce_lr = ReduceLROnPlateau(monitor='loss')
    checkpoint = ModelCheckpoint(args.save_path,  save_best_only=True, save_weights_only=True, save_freq='epoch')
    tb = TensorBoard(args.tens_path)
    cbs = [er, reduce_lr, checkpoint]
    return model.fit(dataset, callbacks = cbs, epochs = EPOCHS, verbose = 1)



def main(args = sys.argv[1:]):
    args = parseArguments()
    data = DataLoader().load_batch()
    model = initDeepNet()
    train(model, data, args)
    model.plot_history(history)

if __name__ == "__main__":
    main()

