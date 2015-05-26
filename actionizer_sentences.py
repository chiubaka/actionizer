#! /usr/bin/python

import numpy as np
import os
import re
from sklearn import datasets, cross_validation
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

MESSAGES_DIR = "data/messages/"
MESSAGE_FILENAME_FORMAT = "msg-%d.txt"
JUDGMENTS_PATH = "data/judgments/judgments.txt"

def load_messages():
    messages = []

    # Read message files in numeric order. os.listdir() returns them sorted by string, not message
    # number.
    filenames = os.listdir(MESSAGES_DIR)
    num_messages = len(filenames)
    for i in range(num_messages):
        filename = MESSAGE_FILENAME_FORMAT % i
        with open(MESSAGES_DIR + filename) as message_file:
            messages.append(message_file.read())

    return messages

def load_sentence_judgments():
    judgments = []
    with open(JUDGMENTS_PATH) as judgments_file:
        for line in judgments_file:
            judgments.append([int(x) for x in line.split()[2:] if len(line.split()) > 2])

    return judgments

def load_sentences():
    messages = load_messages()
    judgments = load_sentence_judgments()
    action_sentences = []
    no_action_sentences = []
    for i in range(len(messages)):
        message = messages[i]
        sentences = parse_sentences(message)
        action_indices = judgments[i]
        if len(action_indices) > 0:
            for i in range(0, len(action_indices), 2):
                start_index = action_indices[i]
                length = action_indices[i+1]
                stop_index = start_index + length
                action_sentence = message[start_index:stop_index].strip().replace('\n', ' ')
                if action_sentence in sentences:
                    action_sentences.append(action_sentence)
                    sentences.remove(action_sentence)
        no_action_sentences.extend(sentences)

    target = [1 for _ in action_sentences]
    target.extend([0 for _ in no_action_sentences])
    action_sentences.extend(no_action_sentences)
    return action_sentences, target

def parse_sentences(message):
    # Split the sentence on periods, exclamation marks, and double newlines. Recombine punctuation
    # marks with their sentences.
    sentences = reduce(lambda acc, elem: acc[:-1] + [acc[-1] + elem] \
        if elem == '.' or elem == '?' or elem == '!' \
        else acc + [elem], re.split(r'([\.\!\?]|\n\n)', message), [])

    # Strip sentences of extra white space.
    # Replace internal newlines with spaces so that newlines don't trip up sklearn tokenizers.
    # Remove all sentences that have length 0 or are completely comprised of whitespace.
    # Remove any sentence starting with the 'From:' header, which should remove the From:, To:, 
    # and Subject:
    sentences = [s.strip().replace('\n', ' ') for s in sentences if len(s) > 0 and not s.isspace() and not s.startswith('From:')]

    return sentences

# Transformer to transform a sparse matrix into a dense matrix for use in an sklearn pipeline.
class DenseTransformer(TransformerMixin):
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

def main():
    sentences, target = load_sentences()

    pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3))), ('to_dense', DenseTransformer()), ('clf', GaussianNB())])
    pipeline.fit(sentences, target)

    scores = cross_validation.cross_val_score(pipeline, sentences, target, scoring='f1', cv=5)
    print "F1: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

    scores = cross_validation.cross_val_score(pipeline, sentences, target, cv=5)
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

if __name__ == "__main__":
    main()
