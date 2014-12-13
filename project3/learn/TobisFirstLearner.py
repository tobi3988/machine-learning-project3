from project3.BagOfWords.bag_of_words import BagOfWords
from project3.learn.CrossValidation import CrossValidation

__author__ = 'tobi'

import unittest
import gc
import threading
import time
import numpy as np
from sklearn import svm


class EveryWordOneFeature(object):
    def __init__(self, slack=1, gamma=1):
        self.slack = slack
        self.gamma = gamma
        self.kernelType = 'linear'
        self.data = np.ones((1000, 1000))
        self.cityClassifier = svm.SVC(kernel=self.kernelType, C=self.slack, gamma=self.gamma, probability=False,
                                      cache_size=1000)
        self.countryClassifier = svm.SVC(kernel=self.kernelType, C=self.slack, gamma=self.gamma, probability=False,
                                         cache_size=1000)
        self.bag = None
        self.numberOfWords = 0

    def fit_cities(self):
        print "Start Fitting cities"
        start = time.time()
        self.cityClassifier.fit(self.fitting_data[:, :self.numberOfWords],
                                self.fitting_data[:, self.numberOfWords])
        end = time.time()
        print "Finished Fitting cities in " + str((end - start)) + "s"

    def fit_countries(self):
        print "Start Fitting countries"
        start = time.time()
        self.countryClassifier.fit(self.fitting_data[:, :self.numberOfWords],
                                   self.fitting_data[:, (self.numberOfWords + 1)])
        end = time.time()
        print "Finished fitting countries in " + str((end - start)) + "s"


    def preprocessing(self, data):
        startOfPreprocessing = time.time()
        print "Start Preprocessing"
        lengthOfTrainingData = self.data.shape[0]
        print "length of trainingData = " + str(lengthOfTrainingData)
        self.bag = BagOfWords(data)
        self.fitting_data = self.bag.get_features_and_labels()
        self.numberOfWords = self.fitting_data.shape[1] - 2
        startOfFittingCities = time.time()
        print "Finished Preprocessing in " + str((startOfFittingCities - startOfPreprocessing)) + "s"

    def fit(self, data):
        self.data = data
        self.preprocessing(data)

        t1 = threading.Thread(target=self.fit_cities)
        t2 = threading.Thread(target=self.fit_countries)
        t1.start()

        t2.start()
        t1.join()
        t2.join()


    def predict_cities(self):
        print "Start predict cities"
        start = time.time()
        self.cityClassifier.predict(self.predict_data[:, :self.numberOfWords])
        end = time.time()
        print "Finished predicting cities in " + str((end - start)) + "s"
        self.cityPrediction

    def predict_countries(self):
        start = time.time()
        print "start predicting countries"
        self.countryClassifier.predict(self.predict_data[:, :self.numberOfWords])
        end = time.time()
        print "finished predicting countries in " + str((end - start)) + "s"
        self.countryPrediction

    def predict(self, predict):
        self.predict_data = self.bag.get_get_validation_features(predict)
        t1 = threading.Thread(target=self.predict_cities)
        t2 = threading.Thread(target=self.predict_countries)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        prediction = np.vstack((self.cityPrediction, self.countryPrediction)).T
        return prediction


class MyTestCase(unittest.TestCase):
    def test_something(self):
        predictor = EveryWordOneFeature()
        crossval = CrossValidation()
        print crossval.run(predictor, 10)


if __name__ == '__main__':
    unittest.main()
