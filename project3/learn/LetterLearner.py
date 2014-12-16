from project3.BagOfWords.LetterFeatures import LetterFeatures
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
    def preprocess_training_data(self, data):
        features = LetterFeatures().get(data[:, 0]).astype(int)
        self.numberOfFeatures = features.shape[1]
        self.fitting_data = np.hstack((features, data[:, 1:].astype(int)))

    def preprocess_predict_data(self, predict):
        self.predict_data = LetterFeatures().get(predict).astype(int)
        self.numberOfFeatures = self.predict_data.shape[1]
        number_of_cities_to_predict = self.predict_data.shape[0]
        self.cityPrediction = np.zeros(number_of_cities_to_predict)
        print self.numberOfFeatures


class MyTestCase(unittest.TestCase):
    def test_letter(self):
        predictor = LetterLearner(kernelType="rbf")
        crossval = CrossValidation()
        print "CrossVal result: " + str(crossval.run(predictor, 10))


if __name__ == '__main__':
    unittest.main()

