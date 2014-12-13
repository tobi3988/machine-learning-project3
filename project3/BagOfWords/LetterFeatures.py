__author__ = 'tobi'

import unittest

import numpy as np


class LetterFeatures(object):

    def count_letters(self, feature, sentence):
        feature[0, 0] = sentence.count("a")
        feature[0, 1] = sentence.count("b")
        feature[0, 2] = sentence.count("c")
        feature[0, 3] = sentence.count("d")
        feature[0, 4] = sentence.count("e")
        feature[0, 5] = sentence.count("f")
        feature[0, 6] = sentence.count("g")
        feature[0, 7] = sentence.count("h")
        feature[0, 8] = sentence.count("i")
        feature[0, 9] = sentence.count("j")
        feature[0, 10] = sentence.count("k")
        feature[0, 11] = sentence.count("l")
        feature[0, 12] = sentence.count("m")
        feature[0, 13] = sentence.count("n")
        feature[0, 14] = sentence.count("o")
        feature[0, 15] = sentence.count("p")
        feature[0, 16] = sentence.count("q")
        feature[0, 17] = sentence.count("r")
        feature[0, 18] = sentence.count("s")
        feature[0, 19] = sentence.count("t")
        feature[0, 20] = sentence.count("u")
        feature[0, 21] = sentence.count("v")
        feature[0, 22] = sentence.count("w")
        feature[0, 23] = sentence.count("x")
        feature[0, 24] = sentence.count("y")
        feature[0, 25] = sentence.count("z")

    def get_from_string(self, sentence):
        sentence = sentence.lower()
        feature = np.zeros((1, 26))
        if sentence.__len__() > 0:
            self.count_letters(feature, sentence)
        return feature


class TestLetterFeatures(unittest.TestCase):
    def test_empty_string(self):
        features = LetterFeatures()
        actual = features.get_from_string("")
        expected = np.zeros((1, 26))
        self.assertArrayEqual(actual, expected)

    def test_one_a(self):
        features = LetterFeatures()
        actual = features.get_from_string("a")
        expected = np.zeros((1, 26))
        expected[0, 0] = 1
        self.assertArrayEqual(actual, expected)

    def test_two_a(self):
        features = LetterFeatures()
        actual = features.get_from_string("aa")
        expected = np.zeros((1,26))
        expected[0,0] = 2
        self.assertArrayEqual(actual,expected)

    def test_case_insensitive(self):
        features = LetterFeatures()
        actual = features.get_from_string("A")
        expected = np.zeros((1, 26))
        expected[0, 0] = 1
        self.assertArrayEqual(actual, expected)

    def test_one_b(self):
        features = LetterFeatures()
        actual = features.get_from_string("b")
        expected = np.zeros((1,26))
        expected[0, 1] = 1
        self.assertArrayEqual(actual,expected)

    def test_one_alphabet(self):
        features = LetterFeatures()
        actual = features.get_from_string("abcdefghijklmnopqrstuvwxyz")
        expected = np.ones((1,26))
        self.assertArrayEqual(actual,expected)


    def assertArrayEqual(self, actual, expected):
        self.assertTrue(np.array_equal(actual, expected), "Features are not what expected")


if __name__ == '__main__':
    unittest.main()
