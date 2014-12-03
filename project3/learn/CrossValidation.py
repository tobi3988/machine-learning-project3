__author__ = 'tobi'

import unittest
import numpy as np
import csv
from sklearn.cross_validation import KFold

class CrossValidation(object):

    def run(self, predictor, folds, csv_name = "training.csv"):
        trainingData = self.import_csv(csv_name)
        return self.runWithProvidedDataset(predictor, folds, trainingData)


    def import_csv(self, csvName):
        reader = csv.reader(open(csvName, "rb"), delimiter=',')
        x = list(reader)
        return np.array(x)

    def runWithProvidedDataset(self, predictor, folds, trainingData):
        kf = KFold(trainingData.shape[0], n_folds=folds, shuffle=True)
        totalScores = 0
        for train, test in kf:
            data_train, data_test = trainingData[train], trainingData[test]
            predictor.fit(data_test)
            X_test = data_test[:, 0]
            y_test = data_test[:, 1:]
            y_predict = predictor.predict(X_test)

            scores = self.calculateScores(y_predict, y_test)
            totalScores += scores
        avgScore = totalScores / folds
        return avgScore

    def calculateScores(self, predicted, test):
        difference = predicted.astype(int) - test.astype(int)
        cityCodes = difference[:, 0]
        countryCodes = difference[:, 1]
        lengthOfTest = test.shape[0]
        numberOfFalseCitys= sum(cityCodes != 0)
        numberOfFalseCountrys = sum(countryCodes != 0)
        score = (numberOfFalseCitys + 0.25*numberOfFalseCountrys)/ lengthOfTest
        return score


class CrossValidationTest(unittest.TestCase):
    def test_cross_validation(self):
        CrossValidation().run(MockPredictor(), 10)

class MockPredictor(object):

    def fit(self, trainingData):
        pass

    def predict(self, predictionData):
        numberOfSamples = predictionData.shape[0]
        return np.zeros((numberOfSamples, 2))

        


if __name__ == '__main__':
    unittest.main()
