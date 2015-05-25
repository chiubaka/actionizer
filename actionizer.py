#! /usr/bin/python

import numpy as np
import os
from sklearn import datasets

MESSAGES_DIR = "data/messages/"
JUDGMENTS_PATH = "data/judgments/judgments.txt"

def load_messages():
    messages = []
    for filename in os.listdir(MESSAGES_DIR):
        with open(MESSAGES_DIR + filename) as message_file:
            messages.append(message_file.read())

    return messages

def tfidf(documents):
    # TODO: Stub implementation
    return [[]]

def load_judgments():
    judgments = []
    with open(JUDGMENTS_PATH) as judgments_file:
        for line in judgments_file:
            judgments.append(1 if len(line.split()) > 2 else 0)

    return judgments

def main():
    messages = load_messages()
    target = load_judgments()
    print target
    data = tfidf(messages)

if __name__ == "__main__":
    main()
