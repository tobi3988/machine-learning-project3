from project3.BagOfWords.bag_of_words import BagOfWords
from project3.learn.CrossValidation import CrossValidation

__author__ = 'tobi'

import unittest
import gc
import threading
import time
import numpy as np
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier


class EveryWordOneFeature(object):
    def __init__(self, slack=1, gamma=1, kernelType='linear'):
        self.slack = slack
        self.gamma = gamma
        self.kernelType = kernelType
        self.data = np.ones((1000, 1000))
        self.cityClassifier = {}
        self.countryClassifier = OneVsOneClassifier(
            svm.SVC(kernel=self.kernelType, C=self.slack, gamma=self.gamma, probability=False,
                    cache_size=1000))
        self.bag = None
        self.numberOfFeatures = 0
        #Features and labels
        self.fitting_data = None
        self.predict_data = None
        self.cityPrediction = {}
        self.countryPrediction = None

    def fit_cities(self, trainingData, cityCode):
        print "Start fitting cities for country "  + str(cityCode)
        self.cityClassifier[cityCode] = OneVsOneClassifier(
            svm.SVC(kernel=self.kernelType, C=self.slack, gamma=self.gamma, probability=False))
        start = time.time()
        self.cityClassifier[cityCode].fit(trainingData[:, :self.numberOfFeatures],
                                trainingData[:, self.numberOfFeatures])
        end = time.time()
        print "Finished fitting cities in " + str((end - start)) + "s"

    def fit_countries(self):
        print "Start Fitting countries"
        start = time.time()
        self.countryClassifier.fit(self.fitting_data[:, :self.numberOfFeatures],
                                   self.fitting_data[:, (self.numberOfFeatures + 1)])
        end = time.time()
        print "Finished fitting countries in " + str((end - start)) + "s"


    def preprocess_training_data(self, data):
        startOfPreprocessing = time.time()
        print "Start Preprocessing"
        lengthOfTrainingData = self.data.shape[0]
        print "length of trainingData = " + str(lengthOfTrainingData)
        self.bag = BagOfWords(data)
        self.fitting_data = self.bag.get_features_and_labels()
        self.numberOfFeatures = self.fitting_data.shape[1] - 2
        startOfFittingCities = time.time()
        print "Finished Preprocessing in " + str((startOfFittingCities - startOfPreprocessing)) + "s"

    def fit(self, data):
        self.data = data
        self.preprocess_training_data(data)
        self.numberOfFeatures = self.fitting_data.shape[1] - 2

        self.fit_countries()
        #t1 = threading.Thread(target=self.fit_cities)
        #t2 = threading.Thread(target=self.fit_countries)

        # t1 = threading.Thread(target=self.fit_cities)
        cityCodes = np.unique(self.fitting_data[:, self.numberOfFeatures + 1])
        for cityCode in cityCodes:
            cityCode
            countryIndices = np.where(self.fitting_data[:, self.numberOfFeatures + 1] == cityCode)[0]
            self.fit_cities(self.fitting_data[countryIndices], cityCode)

        #t1.start()
        #t2.start()
        #t1.join()
        #t2.join()


    def predict_cities(self,data,cityCode):
        print "Start predict cities"
        start = time.time()
        print data[:, :self.numberOfFeatures]
        self.cityPrediction[cityCode] = self.cityClassifier[cityCode].predict(data[:, :self.numberOfFeatures])
        end = time.time()
        print "Finished predicting cities in " + str((end - start)) + "s"


    def predict_countries(self):
        start = time.time()
        print "start predicting countries"
        print self.predict_data
        self.countryPrediction = self.countryClassifier.predict(self.predict_data[:, :self.numberOfFeatures])
        end = time.time()
        print "finished predicting countries in " + str((end - start)) + "s"


    def preprocess_predict_data(self, predict):
        self.predict_data = self.bag.get_get_validation_features(predict)

    def predict(self, predict):
        self.preprocess_predict_data(predict)
        self.numberOfFeatures = self.predict_data.shape[1]
        # t1 = threading.Thread(target=self.predict_cities)
        self.predict_countries()
        self.countryPrediction
        cityCodes = np.unique(self.countryPrediction)
        joinedCityPredictions = np.zeros(predict.shape[0])
        for cityCode in cityCodes:
            cityCode
            countryIndices = np.where(self.countryPrediction == cityCode)[0]
            self.predict_cities(self.predict_data[countryIndices], cityCode)
            joinedCityPredictions[countryIndices] = self.cityPrediction[cityCode]
        #t1.start()
        #t2.start()
        #t1.join()
        #t2.join()
        print joinedCityPredictions
        prediction = np.vstack((joinedCityPredictions, self.countryPrediction)).T
        return prediction


class MyTestCase(unittest.TestCase):
    def test_something(self):
        predictor = EveryWordOneFeature()
        crossval = CrossValidation()
        print crossval.run(predictor, 10)


if __name__ == '__main__':
    unittest.main()
