from project3.BagOfWords.bag_of_words import BagOfWords

__author__ = 'tobi'

import unittest
import numpy as np
import csv


class Learn(object):

    def import_csv(self, csvName):
        reader = csv.reader(open(csvName, "rb"), delimiter=',')
        x = list(reader)
        importedData = np.array(x)
        bag = BagOfWords(importedData)
        np.savetxt("trainingfeatures.csv", bag.get_features_and_labels(), delimiter=",")


class MyTestCase(unittest.TestCase):
    def test_import(self):
        learn = Learn()
        learn.import_csv("training.csv")


if __name__ == '__main__':
    unittest.main()
