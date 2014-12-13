from project3.BagOfWords.bag_of_words import BagOfWords
from project3.learn.CrossValidation import CrossValidation

__author__ = 'tobi'

import unittest
import gc
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

    def fit(self, data):
        self.data = data


    def predict(self, predict):
        startOfPreprocessing = time.time()
        print "Start Preprocessing"
        lengthOfPredictData = predict.shape[0]
        lengthOfTrainingData = self.data.shape[0]
        print "length of trainingData = " + str(lengthOfTrainingData)
        predict = np.hstack((predict, np.zeros((lengthOfPredictData, 2)).astype(int)))
        trainingAndPredictData = np.vstack((self.data, predict))
        del self.data
        gc.collect()
        bag = BagOfWords(trainingAndPredictData)
        transformed_data = bag.get_features_and_labels()
        numberOfWords = transformed_data.shape[1] - 2
        startOfFittingCities = time.time()
        print "Finished Preprocessing in " + str((startOfFittingCities - startOfPreprocessing)) + "s"
        print "Starting Fitting cities"
        self.cityClassifier.fit(transformed_data[:lengthOfTrainingData, :numberOfWords],
                                transformed_data[:lengthOfTrainingData, numberOfWords])
        startOfFittingCountries = time.time()
        print "Finished Fitting cities in " + str((startOfFittingCountries - startOfFittingCities)) + "s"
        print "Starting Fitting countries"
        self.countryClassifier.fit(transformed_data[:lengthOfTrainingData, :numberOfWords],
                                   transformed_data[:lengthOfTrainingData, (numberOfWords + 1)])
        startOfPredictingCities = time.time()
        print "Finished fittign countries in " + str((startOfPredictingCities - startOfFittingCountries)) + "s"
        print "Start predict cities"
        cityPrediction = self.cityClassifier.predict(transformed_data[lengthOfTrainingData:, :numberOfWords])
        startOfPredictingCountries = time.time()
        print "Finished predicting cities in " + str((startOfPredictingCountries - startOfPredictingCities)) + "s"
        print "start predicting countries"
        countryPrediction = self.countryClassifier.predict(transformed_data[lengthOfTrainingData:, :numberOfWords])
        endOfPredictingCountries = time.time()
        print "finished predicting countries in " + str((endOfPredictingCountries - startOfPredictingCountries)) + "s"
        prediction = np.vstack((cityPrediction, countryPrediction)).T
        return prediction
        # return cityPrediction


class MyTestCase(unittest.TestCase):
    def test_something(self):
        predictor = EveryWordOneFeature()
        crossval = CrossValidation()
        print crossval.run(predictor, 10)


if __name__ == '__main__':
    unittest.main()
