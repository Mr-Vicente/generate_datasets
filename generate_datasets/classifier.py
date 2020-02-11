#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:46:48 2020
"""

"""
    Classifier
    
    Frederico Vicente, NOVA FCT, MIEI
    Ludwig Krippahl
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from sklearn.model_selection import KFold

import numpy as np
import matplotlib.pyplot as plt

fmnist = tf.keras.datasets.fashion_mnist

class Classifier():
    def __init__(self):
        # Data
        self.train_dataset = None
        self.test_dataset = None
        self.train_labels = None
        self.test_labels = None

    def load_dataset(self,dataset):
        # Load dataset
        (train_images, train_labels), (test_images, test_labels) = dataset

        # Reshape
        self.train_dataset = train_images.reshape(train_images.shape[0], 28, 28, 1)
        self.test_dataset = test_images.reshape(test_images.shape[0], 28, 28, 1)

        # Convert to one-hot encoding
        self.train_labels = to_categorical(train_labels)
        self.test_labels = to_categorical(test_labels)

        self.prepare_data()

    def prepare_data(self):
        # Convert from ints to floats
        self.train_dataset = self.train_dataset.astype("float32")
        self.test_dataset = self.test_dataset.astype("float32")

        # Normalize to range 0-1
        self.train_dataset = self.train_dataset / 255.0
        self.test_dataset = self.test_dataset / 255.0

    def build_model(self):
        # Model Architecture
        model = Sequential()
        model.add(Conv2D(filters=64 , kernel_size=(3,3),padding='same', activation='relu',kernel_initializer='he_uniform', input_shape=(28,28,1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(units=100, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(units=10, activation='softmax'))

        optimizer = SGD(learning_rate=0.01, momentum=0.9)
        # compile model
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def evaluate_model(self, n_folds=5):
        scores, histories = list(), list()
        # Prepare cross validation
        kfold = KFold(n_folds, shuffle=True, random_state=1)
        # Enumerate splits
        for train_ix, test_ix in kfold.split(self.train_dataset):
            model = self.build_model()
            # select rows for train and test
            trainX, trainY, testX, testY = self.train_dataset[train_ix], self.train_labels[train_ix], self.train_dataset[test_ix], self.train_labels[test_ix]
            # fit model
            history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
            # evaluate model
            _, acc = model.evaluate(testX, testY, verbose=0)
            print('> %.3f' % (acc * 100.0))
            # append scores
            scores.append(acc)
            histories.append(history)
        self.summarize_diagnostics(histories)
        self.summarize_performance(scores)

    
    # plot diagnostic learning curves
    def summarize_diagnostics(self,histories):
        for i in range(len(histories)):
            # plot loss
            plt.subplot(211)
            plt.title('Cross Entropy Loss')
            plt.plot(histories[i].history['loss'], color='blue', label='train')
            plt.plot(histories[i].history['val_loss'], color='orange', label='test')
            # plot accuracy
            plt.subplot(212)
            plt.title('Classification Accuracy')
            plt.plot(histories[i].history['accuracy'], color='blue', label='train')
            plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
        plt.show()

    # summarize model performance
    def summarize_performance(self,scores):
        # print summary
        print('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores)*100, np.std(scores)*100, len(scores)))
        # box and whisker plots of results
        plt.boxplot(scores)
        plt.show()
        
    def train_model(self):
        model = self.build_model()
        model.fit(self.train_dataset, self.train_labels, epochs=10, batch_size=32, validation_data=(self.test_dataset, self.test_labels), verbose=0)
        model.save('classifier.h5')
        
    def predict(self,image):
        model = load_model('classifier.h5')
        result = model.predict_classes(image)
        print(result[0])
        
