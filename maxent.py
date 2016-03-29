# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
from corpus import NamesCorpus, ReviewCorpus
import numpy as np
from random import shuffle
import math
from scipy import misc
import string
import re

class MaxEnt(Classifier):
    def __init__(self):
        self.features = {}         #maps feature string to index
        self.reverse_features = {} #maps feature's index to feature string
        self.labels = {}
        self.params = None
        self.my_labels = {}
        self.their_labels = {}
        self.punctuation = re.compile('[%s]' % re.escape(string.punctuation))

    def get_model(self): return None

    def set_model(self, model): pass

    model = property(get_model, set_model)

    def train(self, instances, dev_instances):
        print 'Training MaxEnt Classifier...'
        """
        determine features and labels in training instances
        create parameter matrix using these labels and features
        """
        row = index = 0
        for item in instances:
            """determine label index if not already seen"""
            if item.label not in self.labels: 
                self.labels[item.label] = row
                row += 1
            if 'names' in item.source:
                item.feature_vector = self.name_features(item.data)
            elif 'review' in item.source:
                data = item.data.lower().split()
                item.feature_vector = self.ngrams(data, 2)
            else: print 'Error: something wrong with input instances'

            """match features with an index to index into matrices later"""
            for feature in item.feature_vector:
                if feature not in self.features:
                    self.features[feature] = index
                    index += 1

        """rows are labels, columns are features"""
        self.params = np.zeros(shape=(len(self.labels), len(self.features)))

        self.train_sgd(instances, dev_instances, 0.0001, 100)

    def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size):
        print 'Running Mini-batch Stochastic Gradient...'
        """Train MaxEnt model with Mini-batch Stochastic Gradient """
        '''get the features out of each dev_instance'''
        for instance in dev_instances:
            if 'names' in instance.source:
                instance.feature_vector = self.name_features(instance.data)
            elif 'review' in instance.source:
                data = instance.data.lower().split()
                """data = [w for w in data if w not in stopwords.words('english')]"""
                instance.feature_vector = self.ngrams(data, 2)  #not a set() here

        likelihood = 0
        change = 1
        count = 0
        accuracy = 0
        print 'Likelihood         Accuracy'
        """do mini batch thing"""
        while change > learning_rate:
            shuffle(train_instances)
            batch = train_instances[:batch_size]
            observed = np.zeros(shape=(len(self.labels), len(self.features)))
            expected = np.zeros(shape=(len(self.labels), len(self.features)))
            gradient = np.zeros(shape=(len(self.labels), len(self.features)))
            """generate observed and expected counts"""
            for instance in batch:
                label_index = self.labels[instance.label]
                for feature in instance.feature_vector:
                    if feature in self.features:
                        feature_index = self.features[feature]
                        observed[label_index, feature_index] += 1
                        expected[label_index, feature_index] = self.expected_prob(feature, instance.label)
            """compute gradient"""
            gradient = (observed - expected) * learning_rate
            self.params += gradient
            likelihood = self.ave_log_likelihood(dev_instances)
            new_accuracy = self.compute_accuracy(dev_instances)
            change = abs(accuracy - new_accuracy)    #likelihood should always decrease
            accuracy = new_accuracy
            print str(likelihood) + "     " + str(accuracy)

    def expected_prob(self, feature, label):
        feature_index = self.features[feature]
        label_index = self.labels[label]
        num = math.exp(self.params[label_index, feature_index])
        denom = sum([math.exp(p) for p in self.params[:,feature_index]])
        return num/denom

    def name_features(self, data):
        first_letter = '|'.join([data[0], 'f'])
        last_letter = '|'.join([data[-1], 'l'])
        return [first_letter, last_letter]

    def ave_log_likelihood(self, batch):
        likelihoods = []
        for instance in batch:
            num = denom = float(0)
            label_index = self.labels[instance.label]
            for feature in instance.feature_vector:
                if feature in self.features:
                    feature_index = self.features[feature]
                    num += self.params[label_index, feature_index]
                    denom += sum([math.exp(p) for p in self.params[:,feature_index]])
                    denom -= math.exp(self.params[label_index, feature_index]) #don't include param feature for instance's label in denom
            if denom > 0:
                likelihoods.append(math.log(math.exp(num)/denom))

        """return the average likelihood for the batch"""
        return sum(likelihoods)/len(likelihoods) * -1

    def ngrams(self, input, n):
        output = []
        for i in range(len(input) - n + 1):
            ngram = ' '.join(input[i : i + n])
            self.punctuation.sub('', ngram)
            output.append(ngram)
        return output


    def classify(self, instance):
        if 'names' in instance.source:
            instance.feature_vector = self.name_features(instance.data)
        elif 'review' in instance.source:
            data = instance.data.lower().split()
            instance.feature_vector = self.ngrams(data, 2)
        max_prob = 0
        instance_label = ''
        instance_features = []
        for feature in instance.feature_vector:
            if feature in self.features:
                """keep track of which features the insatnce has to index later"""
                instance_features.append(self.features[feature])
            for label in self.labels:
                label_index = self.labels[label]
                tmp_prob = math.exp(sum([self.params[label_index, index] for index in instance_features]))
                if tmp_prob >= max_prob:
                    max_prob = tmp_prob
                    instance_label = label
        self.my_labels[instance_label] = self.my_labels.get(instance_label, 1) + 1
        self.their_labels[instance.label] = self.their_labels.get(instance.label, 1) + 1
        return instance_label

    def compute_accuracy(self, dev_instances):
        correct = [self.classify(x) == x.label for x in dev_instances]
        return float(sum(correct)) / len(correct)
    
