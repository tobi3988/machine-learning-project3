from project3.learn.NGramLearner import NGramLearner
from project3.learn.TobisFirstLearner import EveryWordOneFeature

__author__ = 'tobi'

import unittest

import numpy as np
import csv


class MyTestCase(unittest.TestCase):

    def makePrediction(self):
        trainingData = self.import_csv("training.csv")
        print trainingData
        predictor = NGramLearner()
        print "fit"
        predictor.fit(trainingData)

        validationData = self.import_csv("validation.csv")

        print "predict"
        print validationData
        validationResults = predictor.predict(validationData)
        np.savetxt("validationresult.csv", validationResults, delimiter=",")



    def import_csv(self, csvName):
        reader = csv.reader(open(csvName, "rb"), delimiter=',')
        x = list(reader)
        return np.array(x)


    def test_something(self):
        self.makePrediction()


if __name__ == '__main__':
    unittest.main()
