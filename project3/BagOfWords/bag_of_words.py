__author__ = 'tobi'

import unittest
import numpy as np


class BagOfWords(object):
    def __init__(self, samples):
        self.samples = samples
        self.numberOfSamples = self.samples.shape[0]
        self.sentences = self.samples[:, 0]
        self.labels = self.samples[:, 1:].astype(int)
        self.dictionary = self.create_dictionary()
        self.numberOfWords = self.dictionary.keys().__len__()
        self.features = np.hstack((np.zeros((self.numberOfSamples, self.numberOfWords)).astype(int), self.labels))
        print "allocated feature array"
        self.create_features()

    def create_dictionary(self):
        self.dictionary = {}
        index = 0
        for sentence in self.sentences:
            self.extract_words(index, sentence.lower())
            index += 1
        return self.dictionary

    def get_features_and_labels(self):

        return self.join_features_and_labels()

    def join_features_and_labels(self):
        return self.features

    def create_features(self):
        index = 0
        for word in sorted(self.dictionary):
            self.create_feature_for_word(word, index)
            index += 1

    def create_feature_for_word(self, word, wordIndex):
        for index in self.dictionary[word]:
            self.features[index, wordIndex] += 1

    def add_or_extend_word(self, index, word):
        if word in self.dictionary:
            self.dictionary[word].append(index)
        else:
            self.dictionary[word] = [index]

    def extract_words(self, index, sentence):
        words = str(sentence).split()
        for word in words:
            self.add_or_extend_word(index, word)

    def get_features(self):
        return self.features[:, :self.numberOfWords]

    def get_labels(self):
        return self.features[:, self.numberOfWords:]

    def get_validation_features(self, validationSet):
        for word in self.dictionary:
            self.dictionary[word] = []




class BagOfWordsTest(unittest.TestCase):
    def test_one_word(self):
        bag = BagOfWords(np.array([["word", 1, 2]]))
        expectedFeatures = np.array([[1, 1, 2]])
        actualFeatures = bag.get_features_and_labels()
        self.assertArrayEqual(actualFeatures, expectedFeatures)

    def testOtherWord(self):
        bag = BagOfWords(np.array([["another", 1, 2]]))
        expected = np.array([[1, 1, 2]])
        actual = bag.get_features_and_labels()
        self.assertArrayEqual(actual, expected)

    def test_other_city_and_country_code(self):
        bag = BagOfWords(np.array([["word", 2, 1]]))
        expected = np.array([[1, 2, 1]])
        actual = bag.get_features_and_labels()
        self.assertArrayEqual(actual, expected)

    def test_two_samples(self):
        bag = BagOfWords(np.array([["word", 1, 2], ["word", 1, 2]]))
        expected = np.array([[1, 1, 2], [1, 1, 2]])
        actual = bag.get_features_and_labels()
        self.assertArrayEqual(actual, expected)

    def test_two_words(self):
        bag = BagOfWords(np.array([["two words", 1, 2]]))
        expected = np.array([[1, 1, 1, 2]])
        actual = bag.get_features_and_labels()
        self.assertArrayEqual(actual, expected)

    def test_two_words_two_samples(self):
        bag = BagOfWords(np.array([["two words", 1, 2], ["two words", 1, 2]]))
        expected = np.array([[1, 1, 1, 2], [1, 1, 1, 2]])
        actual = bag.get_features_and_labels()
        self.assertArrayEqual(actual, expected)

    def test_two_samples_with_different_words(self):
        bag = BagOfWords(np.array([["one", 1, 2], ["two", 1, 2]]))
        expected = np.array([[1, 0, 1, 2], [0, 1, 1, 2]])
        actual = bag.get_features_and_labels()
        self.assertArrayEqual(actual, expected)

    def test_two_same_words(self):
        bag = BagOfWords(np.array([["word word", 1, 2]]))
        expected = np.array([[2, 1, 2]])
        actual = bag.get_features_and_labels()
        self.assertArrayEqual(actual, expected)

    def test_lowercase(self):
        bag = BagOfWords(np.array([["WORD word", 1, 2]]))
        expected = np.array([[2, 1, 2]])
        actual = bag.get_features_and_labels()
        self.assertArrayEqual(actual, expected)

    def test_complicated_example(self):
        bag = BagOfWords(np.array([["this is a whole sentence", 2, 5],
                                   ["this is another one is it", 1000, 1989]]))
        expected = np.array([[1, 0, 1, 0, 0, 1, 1, 1, 2, 5],
                             [0, 1, 2, 1, 1, 0, 1, 0, 1000, 1989]])
        actual = bag.get_features_and_labels()
        self.assertArrayEqual(actual, expected)

    def test_get_only_features(self):
        bag = BagOfWords(np.array([["this is a whole sentence", 3, 5]]))
        expected = np.array([[1, 1, 1, 1, 1]])
        actual = bag.get_features()
        self.assertArrayEqual(actual, expected)

    def test_get_only_labels(self):
        bag = BagOfWords(np.array([["this is a whole sentence", 3, 5]]))
        expected = np.array([[3, 5]])
        actual = bag.get_labels()
        self.assertArrayEqual(actual, expected)

    def test_init_only_with_sentences(self):
        bag = BagOfWords(np.array([["this is only a sentence"]]))
        expected = np.array([[1, 1, 1, 1, 1]])
        actual = bag.get_features()
        self.assertArrayEqual(actual, expected)

    def test_get_features_for_validation(self):
        bag = BagOfWords(np.array([["this is only a sentence"]]))
        validationData = np.array([["this is only a only"]])
        expected = np.array([[1,1, 2, 0, 1]])
        actual = bag.get_validation_features(validationData)
        self.assertArrayEqual(actual, expected)


    def assertArrayEqual(self, actual, expected):
        self.assertTrue(np.array_equal(actual, expected), "Features are not what expected")


if __name__ == '__main__':
    unittest.main()
