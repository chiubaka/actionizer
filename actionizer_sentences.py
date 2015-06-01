#! /usr/bin/python

import numpy as np
import os
import random
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

def overlap(start, length, action_start, action_length, threshold):
    min_start = min(start, action_start)
    max_start = max(start, action_start)
    # These names are a little confusing--not the minimum length, but rather the length associated
    # with the min_start. Couldn't think of a better name.
    min_length = length if min_start == start else action_length
    max_length = length if max_start == start else action_length

    # min_start + min_length is the end of the string that comes first in the message. When
    # max_start is subtracted out, we should get the length of overlap in the message. We don't
    # really have a use for negative overlap_length values so just make anything below 0 into a 0.
    # We can't have an overlap that is longer than the length of the action.
    overlap_length = min(max(min_start + min_length - max_start, 0), action_length)

    # If the amount that overlapped was greater than or equal to the threshold, return true. Else
    # return false.
    return overlap_length / float(action_length) >= threshold

def load_sentences():
    messages = load_messages()
    judgments = load_sentence_judgments()
    num_action_indices = 0
    action_sentences = set()
    no_action_sentences = set()
    for i in range(len(messages)):
        message = messages[i]
        sentences = parse_sentences(message)
        action_indices = judgments[i]
        if len(action_indices) > 0:
            for i in range(0, len(action_indices), 2):
                num_action_indices += 1
                action_start = action_indices[i]
                action_length = action_indices[i+1]
                action_sentence = message[action_start:action_start+action_length].replace('\n', ' ')
                action_found = False
                for start, length, sentence in sentences:
                    if overlap(start, length, action_start, action_length, 0.30):
                        action_found = True
                        action_sentences.add((start, length, sentence))
                        
        # Add all elements that are in sentences but not in action_sentences to the
        # no_action_sentences set
        no_action_sentences.update(sentences.difference(action_sentences))

    target = [1 for _ in action_sentences]
    print '# action annotations:', num_action_indices
    print '# action sentences detected:', len(action_sentences)
    target.extend([0 for _ in no_action_sentences])
    sentences = [s[2] for s in action_sentences]
    sentences.extend([s[2] for s in no_action_sentences])
    print '# total sentences detected:', len(sentences)

    combined = zip(target, sentences)
    random.shuffle(combined)
    target[:], sentences[:] = zip(*combined)
    return sentences, target

def parse_sentences(message):
    # Split the sentence on periods, exclamation marks, and double newlines. Recombine punctuation
    # marks with their sentences.
    sentences = reduce(lambda acc, elem: acc[:-1] + [acc[-1] + elem] \
        if elem == '.' or elem == '?' or elem == '!' \
        else acc + [elem], re.split(r'([\.\!\?]|\n\n)', message), [])

    sentences = [s for s in sentences if len(s) > 0 and not s.isspace() and not s.strip().startswith('From:') \
        and not s.strip().startswith('To:') and not s.strip().startswith('Subject:') \
        and not s.strip().startswith('Date:') and not s.strip().startswith('A:')]

    starts = [message.index(s) for s in sentences]
    lengths = [len(s) for s in sentences]

    # Strip sentences of extra white space.
    # Replace internal newlines with spaces so that newlines don't trip up sklearn tokenizers.
    # Remove all sentences that have length 0 or are completely comprised of whitespace.
    # Remove any sentence starting with the 'From:' header, which should remove the From:, To:, 
    # and Subject:
    sentences = [s.strip().replace('\n', ' ') for s in sentences]

    # Return sentences along with their original start location in the message and their original
    # length. This is done so that we can easily compute overlap with action item spans.
    return set(zip(starts, lengths, sentences))

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
