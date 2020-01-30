from nltk.classify.util import apply_features
from nltk import NaiveBayesClassifier
import pandas as pd
import pickle
import collections
from sklearn.model_selection import train_test_split

class SpamClassifier:
    def extract_tokens(self, text, target):
        '''returns array of tuples where each tuple is defined by (tokenized_text, label)
        parameters:
            test: array of tets
            target: array of targets
        Note: Consider only those words which have all alphabets and atleast 3 characters
        '''
        corpus = []

        corpus = [(tokenized_text, label)
                  for tokenized_text, label in zip(z, target)]

        return corpus

    def get_features(self, corpus):
        '''
        returns a Set of unique words in complete corpus.
        parameters - corpus: tokenized corpus along with target labels (i.e.) he output of extract_tokens function

        Return Type is a set
        '''

    def extract_features(self, document):
        '''
        maps each input text into feature vector
        parameters - document - string

        Return Type - A dictionary with keys being the train data set word features.
                        The values correspond to True or False
        '''
        features = {}
        doc_words = set(document)

        # iterate through word_features to find if the doc_words contains it or not

        return features

    def train(self, text, labels):
        '''
        Returns traine model and set of unique words in training data
        also set trained model to 'self.classifier' variable and set of
        unique words to 'self.word_features' variable.
        '''

        # self.corpus = extract_tokens()
        # self.word_features = get_features()
        # train_set = apply_features(self.extract_features, self.corpus)
        # Now train the NaiveBayesClassifier with train_set
        # self.classifier = 

        # return self.classifier, self.word_features

    def predict(self, text):
        '''
        Returns prediction labels of given input text
        Allowed Text can be simple string i.e one input email, a list of emails, or a dictionary of emails 
        '''

        if isinstance(text, list):
            pred = []
            for sentence in list(text):
                pred.append(self.classifier.classify(self.extract_features(sentence.split())))
            return pred

        if isinstance(text, (collections.OrderedDict)):
            pred = collections.OrderedDict()
            for label, sentence in text.items():
                pred[label] = self.classifier.classify(self.extract_features(sentence.split()))
            return pred

        return self.classifier.classify(self.extract_features(text.split()))
    
if __name__ == "__main__":
    data = pd.read_csv('emails.csv')
    train_X, text_X, train_Y, text_Y = train_test_split(data['text'.values, data['spam'].values, text_size=0.25, random_state=50, shuffle= True, stratify=data['spam'.values]])
    classifier = SpamClassifier()
    classifier_model, model_word_features = classifier.train(train_X, train_Y)
    model_name = 'spam_classifier_model.pk'
    model_word_features_name = 'spam_classifier_model_word_features.pk'
    with open(model_name, 'wb') as model_fp:
        pickle.dump(classifier_model, model_fp)
    with open(model_word_features_name, 'wb') as model_fp:
        pickle.dump(model_word_features, model_fp)
    
    print('Done')
