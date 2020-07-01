# -*- coding: utf-8 -*-
"""
    Cartoon Classifier

    Frederico Vicente, NOVA FCT, MIEI
    Ludwig Krippahl
"""

import sys
sys.path.append("..")

from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from Processing import process_cartoon

from sklearn.model_selection import KFold

import numpy as np
import matplotlib.pyplot as plt

class Classifier():
    def __init__(self):
        # Data
        self.train_dataset = None
        self.test_dataset = None
        self.train_labels = None
        self.test_labels = None
        self.n_classes = None

    def load_dataset(self):
        # Load dataset
        
        train_images, train_labels_ = process_cartoon.decode_data_cartoon()
        
        self.n_classes = train_labels_.shape[1]
        n_elements = train_images.shape[0]
        threshold = 0.2
        max_range = int(n_elements * threshold)
        
        self.train_dataset, self.test_dataset = train_images[max_range:], train_images[:max_range]
        self.train_labels, self.test_labels = train_labels_[max_range:], train_labels_[:max_range]

    def prepare_data(self):

        # Normalize to range 0-1
        self.train_dataset /= 255.0
        self.test_dataset /= 255.0


    def build_model(self):
        # Model Architecture
        model = Sequential()
        model.add(Conv2D(filters=32 , kernel_size=(3,3),padding='same', activation='relu', input_shape=(128,128,3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64 , kernel_size=(3,3),padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(filters=128 , kernel_size=(3,3),padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(filters=256 , kernel_size=(3,3),padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(units=256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(units=self.n_classes, activation='softmax'))

        optimizer = Adam(learning_rate=0.0001)
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
            history = model.fit(trainX, trainY, epochs=50, batch_size=32, validation_data=(testX, testY), verbose=1)
            # evaluate model
            _, acc = model.evaluate(testX, testY, verbose=1)
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
            self.plotTraining(histories[i],show=False)
        plt.show()
    
    def plotTraining(self, hist, show=False):
        plt.subplot(211)
        plt.title('Loss')
        plt.plot(hist.history['loss'], color='blue', label='train')
        plt.plot(hist.history['val_loss'], color='orange', label='test')
        # plot accuracy
        plt.subplot(212)
        plt.title('Classification Accuracy')
        plt.plot(hist.history['accuracy'], color='blue', label='train')
        plt.plot(hist.history['val_accuracy'], color='orange', label='test')
        if(show):
            plt.show()

    # summarize model performance
    def summarize_performance(self,scores):
        # print summary
        print('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores)*100, np.std(scores)*100, len(scores)))
        # box and whisker plots of results
        plt.boxplot(scores)
        plt.show()
        
    def train_model(self,filename='cartoon_classifier'):
        model = self.build_model()
        hist = model.fit(self.train_dataset, self.train_labels, epochs=60, batch_size=64, validation_data=(self.test_dataset, self.test_labels), verbose=1)
        self.plotTraining(hist,show=True)
        model.save('{}.h5'.format(filename))
