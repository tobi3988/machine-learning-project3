__author__ = 'johannes'

import unittest
import numpy as np
import csv
import Levenshtein

#For 10-Fold validation:
#
#Result Mock predictor:
#1.25
#Result Levenshtein predictor:
#0.750706083178
#

class Levenshtein_predictor(object):

    def __init__(self):
        self.dic = {}

    def fit(self, samples):
        for line in samples:
            cityName = line[0]
            cityCode = line[1]
            country = line[2]
            if country not in self.dic:
                self.dic[country] = {}
                self.dic[country][cityCode] = []
                self.dic[country][cityCode].append(cityName.lower())
            else:
                if cityCode not in self.dic[country]:
                    self.dic[country][cityCode] = []
                    self.dic[country][cityCode].append(cityName.lower())
                else:
                    self.dic[country][cityCode].append(cityName.lower())

    def predict(self, features):
        result = np.zeros((len(features), 2))
        i = 0
        for feat in features:
            cityCount = {}
            for key in self.dic:
                for key2 in self.dic[key]:
                    nu = 0
                    for nam in self.dic[key][key2]:
                        nu += Levenshtein.ratio(feat, nam)
                        cityCount[key2] = nu/len(self.dic[key][key2])

            city = max(cityCount, key=cityCount.get)
            countryCount = {}
            for key in self.dic:
                if city in self.dic[key]:
                    countryCount[key] = len(self.dic[key][city])

            country = max(countryCount, key=countryCount.get)
            result[i][0] = city
            result[i][1] = country
            i += 1
        return result