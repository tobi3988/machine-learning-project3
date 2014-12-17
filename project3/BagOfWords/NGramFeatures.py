__author__ = 'johannes'

import unittest

import numpy as np


class NGramFeatures(object):
    def __init__(self, gram=2):
        self.gram = gram
        self.featureOptions = []


    def count_grams(self, feature, sentence):
        for i in range(0, len(self.featureOptions)):
            feature[0, i] = sentence.count(self.featureOptions[i])

    def setFeatureOptions(self, data):
        for sentence in data:
            sen = sentence[0].lower()
            for i in range(0, len(sen)-(self.gram-1)):
                if not self.featureOptions.__contains__(sen[i:i+self.gram]):
                    self.featureOptions.append(sen[i:i+self.gram])
        print "Features:"
        print self.featureOptions

    def get_from_string(self, sentence):
        sentence = sentence.lower()
        feature = np.zeros((1, len(self.featureOptions)))
        if sentence.__len__() > 0:
            self.count_grams(feature, sentence)
        return feature

    def get(self, data):
        self.setFeatureOptions(data)
        features = np.zeros((data.shape[0], len(self.featureOptions)))
        index = 0
        for sentence in data:
            features[index] = self.get_from_string(sentence[0])
            index += 1
        return features


class TestNGramFeatures(unittest.TestCase):

    #Works only for gram = 2 !!!
    def test_fancy(self):
        features = NGramFeatures()
        actual = features.get(np.array([["aBcde"], ["ha ha"]]))
        expected = np.ones((2, 7))
        expected[0] = [1, 1, 1, 1, 0, 0, 0]
        expected[1] = [0, 0, 0, 0, 2, 1, 1]
        self.assertArrayEqual(actual, expected)

    def assertArrayEqual(self, actual, expected):
        self.assertTrue(np.array_equal(actual, expected), "Features are not what expected")

if __name__ == '__main__':
    unittest.main()
