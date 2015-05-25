#! /usr/bin/python

import numpy as np
import os
from sklearn import datasets, cross_validation
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

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

def load_judgments():
    judgments = []
    with open(JUDGMENTS_PATH) as judgments_file:
        for line in judgments_file:
            judgments.append(1 if len(line.split()) > 2 else 0)

    return judgments

# Transformer to transform a sparse matrix into a dense matrix for use in an sklearn pipeline.
class DenseTransformer(TransformerMixin):
    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=False):
        return {}

def main():
    messages = load_messages()
    target = load_judgments()


    pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('to_dense', DenseTransformer()), ('clf', SVC())])
    pipeline.fit(messages, target)

    param_grid = {
        'tfidf__norm': ['l1', 'l2', None],
        'tfidf__smooth_idf': [True, False],
        'tfidf__sublinear_tf': [True, False],
        'clf__C': [1, 10, 100, 1000], 
        'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'clf__degree': [2, 3, 4, 5]
    }

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring='f1', n_jobs=-1, verbose=10, cv=5)
    grid_search.fit(messages, target)

    print grid_search.best_params_

    scores = cross_validation.cross_val_score(grid_search, messages, target, scoring='f1', cv=5)
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

if __name__ == "__main__":
    main()
