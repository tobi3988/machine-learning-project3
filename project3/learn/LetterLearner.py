from project3.BagOfWords.bag_of_words import BagOfWords
from project3.learn.CrossValidation import CrossValidation
from project3.learn.TobisFirstLearner import EveryWordOneFeature

__author__ = 'tobi'

import unittest
import gc
import threading
import time
import numpy as np
from sklearn import svm


class LetterLearner(EveryWordOneFeature):

    def fit(self, data):
        self.data = data
        startOfPreprocessing = time.time()
        print "Start Preprocessing"
        lengthOfTrainingData = self.data.shape[0]
        print "length of trainingData = " + str(lengthOfTrainingData)
        self.bag = BagOfWords(data)
        self.fitting_data = self.bag.get_features_and_labels()
        self.numberOfWords = self.fitting_data.shape[1] - 2
        startOfFittingCities = time.time()
        print "Finished Preprocessing in " + str((startOfFittingCities - startOfPreprocessing)) + "s"

        t1 = threading.Thread(target=self.fit_cities)
        t2 = threading.Thread(target=self.fit_countries)
        t1.start()
        startOfFittingCountries = time.time()

        t2.start()
        self.startOfPredictingCities = time.time()
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

