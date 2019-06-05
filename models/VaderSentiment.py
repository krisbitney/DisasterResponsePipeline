import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'stopwords', 'vader_lexicon'])


class VaderSentiment(BaseEstimator, TransformerMixin):
    ''' sentiment analysis using VADER method described in:
        Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
            Sentiment Analysis of Social Media Text. Eighth International Conference on
            Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

        attributes:
            vader (SentimentIntensityAnalyzer) NTLK VADER sentiment analysis class instance
            tokenizer (function) Python function that tokenizes text; used in lieu of default tokenizer function

    '''

    def __init__(self, tokenizer=None):
        self.vader = SentimentIntensityAnalyzer()
        self.tokenizer = tokenizer

    def tokenize(self, text):
        ''' tokenize text data

        :param text: array-like of text/Strings
        :return: list of lists of String words/tokens
        '''
        text = re.sub('[^a-zA-Z0-9]', ' ', text)
        words = word_tokenize(text.lower())
        words = [w for w in words if w not in stopwords.words('english')]
        lemmatizer = WordNetLemmatizer()
        for w in words:
            lemmatizer.lemmatize(w)
            lemmatizer.lemmatize(w, pos='v')
            lemmatizer.lemmatize(w, pos='a')
        return words

    def analyze(self, text):
        ''' analyze text to produce sentiment scores

        :param text: array-like containing tokenized text data
        :return: Pandas Series with three columns representing negative, neutral, and positive sentiment scores
        '''
        if self.tokenizer is None:
            text = ' '.join(self.tokenize(text))
        else:
            text = ' '.join(self.tokenizer(text))
        scores = self.vader.polarity_scores(text)
        return pd.Series([scores['neg'], scores['neu'], scores['pos']])

    def fit(self, X, y=None):
        ''' does nothing but is required for use in sklearn Pipeline

        :param X: array-like containing text data
        :param y: always None
        :return: self instance reference
        '''
        return self

    def transform(self, X, y=None):
        ''' transform data using analyze function

        :param X: array-like containing string data
        :param y: always None
        :return: Pandas DataFrame with three columns representing sentiment scores
        '''
        scores = pd.Series(X).apply(self.analyze)
        scores.columns = ['neg', 'neu', 'pos']
        return scores