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
    def preprocessing(self, data):
        pass



class MyTestCase(unittest.TestCase):
    def test_letter(self):
        predictor = LetterLearner()
        crossval = CrossValidation()
        print crossval.run(predictor, 10)


if __name__ == '__main__':
    unittest.main()

