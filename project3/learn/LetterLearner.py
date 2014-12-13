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
        features = LetterFeatures().get(data[:, 0])
        self.fitting_data = np.hstack((features, data[:,1:].astype(int)))
        print self.fitting_data

    def preprocess_predict_data(self, predict):
        self.predict_data = LetterFeatures().get(predict).astype(int)



class MyTestCase(unittest.TestCase):
    def test_letter(self):
        predictor = LetterLearner()
        crossval = CrossValidation()
        print "CrossVal result: " + str(crossval.run(predictor, 10))


if __name__ == '__main__':
    unittest.main()

