import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.classify.util import apply_features
from nltk import NaiveBayesClassifier
import pandas as pd
import pickle
import collections
from sklearn.model_selection import train_test_split

def extract_tokens(self, text, target):
    '''returns array of tuples where each tuple is defined by (tokenized_text, label)
    parameters:
        text: array of texts
        target: array of targets
    Note: Consider only those words which have all alphabets and atleast 3 characters
    '''
    corpus = []

    corpus = [(tokenized_text, label)
                for tokenized_text, label in zip(text, target)]

    return corpus


if __name__ == "__main__":
    data = pd.read_csv('emails.csv')
    print(data.head())
    X = data['text'].values
    cv = CountVectorizer()
    X = cv.fit_transform(X)
    train_X, test_X, train_Y, test_Y = train_test_split(
        X, data['spam'].values, test_size=0.25, random_state=50, shuffle=True, stratify=data['spam'].values)
    clf = MultinomialNB()
    clf.fit(train_X, train_Y)
    print(clf.score(test_X, test_Y))
