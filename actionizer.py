#! /usr/bin/python

import numpy as np
import os
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

MESSAGES_DIR = "data/messages/"
JUDGMENTS_PATH = "data/judgments/judgments.txt"

def load_messages():
    messages = []
    for filename in os.listdir(MESSAGES_DIR):
        with open(MESSAGES_DIR + filename) as message_file:
            messages.append(message_file.read())

    return messages

def bag_of_words(documents):
    tokens = set()
    # Go through all the documents and collect all of the unique tokens
    for document in documents:
        for token in document.split():
            tokens.add(token)

    # Create an empty dense matrix with one row for every document and one column for every token
    mat = np.zeros((len(documents), len(tokens)))

    # Convert tokens from a set to a list so that we can grab the index of a token from the set
    tokens = list(tokens)

    # Go through all the documents again and count the frequency of each token
    for row, document in enumerate(documents):
        for token in document.split():
            col = tokens.index(token)
            mat[row][col] += 1

    return mat

def tfidf(data):
    # TODO: Stub implementation
    return data

def load_judgments():
    judgments = []
    with open(JUDGMENTS_PATH) as judgments_file:
        for line in judgments_file:
            judgments.append(1 if len(line.split()) > 2 else 0)

    return judgments

def main():
    messages = load_messages()
    target = load_judgments()
    data = bag_of_words(messages)
    print data
    print target
    #data = tfidf(data)

    # Gaussian Naive Bayes Classifier
    gnb = GaussianNB()
    predictions = gnb.fit(data, target).predict(data)
    print classification_report(target, predictions, target_names=["No action", "Action"])

if __name__ == "__main__":
    main()
