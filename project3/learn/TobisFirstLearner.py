from project3.BagOfWords.bag_of_words import BagOfWords
from project3.learn.CrossValidation import CrossValidation

__author__ = 'tobi'

import unittest
import gc
import numpy as np
from sklearn import svm


class EveryWordOneFeature(object):
    def __init__(self, slack=1, gamma=1):
        self.slack = slack
        self.gamma = gamma
        self.kernelType = 'rbf'
        self.data = np.ones((1000, 1000))
        self.cityClassifier = svm.SVC(kernel=self.kernelType, C=self.slack, gamma=self.gamma, probability=False,
                                     cache_size=1000)
        self.countryClassifier = svm.SVC(kernel=self.kernelType, C=self.slack, gamma=self.gamma, probability=False,
                                     cache_size=1000)

    def fit(self, data):
        self.data = data


    def predict(self, predict):
        lengthOfPredictData = predict.shape[0]
        predict = np.hstack((predict, np.zeros((lengthOfPredictData, 2)).astype(int)))
        trainingAndPredictData = np.vstack((self.data, predict))
        del self.data
        gc.collect()
        bag = BagOfWords(trainingAndPredictData)
        transformed_data = bag.get_features_and_labels()
        numberOfWords = transformed_data.shape[1] - 2
        self.cityClassifier.fit(transformed_data[:lengthOfPredictData, :numberOfWords],
                               transformed_data[:lengthOfPredictData, numberOfWords])
        cityPrediction = self.cityClassifier.predict(transformed_data[lengthOfPredictData:, :numberOfWords])
        self.countryClassifier.fit(transformed_data[:lengthOfPredictData, :numberOfWords],
                               transformed_data[:lengthOfPredictData, (numberOfWords+1)])
        countryPrediction = self.countryClassifier.predict(transformed_data[lengthOfPredictData:, :numberOfWords])
        print "------------------------------------------------------------------------------"
        prediction = np.vstack((cityPrediction, countryPrediction)).T
        return prediction
        #return cityPrediction


class MyTestCase(unittest.TestCase):
    def test_something(self):
        predictor = EveryWordOneFeature()
        crossval = CrossValidation()
        print crossval.run(predictor, 10)


if __name__ == '__main__':
    unittest.main()
