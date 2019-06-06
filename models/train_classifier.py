from VaderSentiment import VaderSentiment

import sys
import re
import joblib

import pandas as pd
from sqlalchemy import create_engine

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    ''' load data from SQL database

    :param database_filepath: String file path/name of SQL database
    :return: tuple containing list of text data values (X), Pandas DataFrame targets/labels (Y), and list of String label names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('etl', engine)
    X = df.message.values
    Y = df.iloc[:, 4:].copy()
    label_names = Y.columns.tolist()
    return X, Y, label_names


def tokenize(text):
    ''' tokenize text data

    :param text: list of text/Strings
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


def build_model():
    ''' build Linear Support Vector Machine prediction model:
        transforms list of raw text entries using "bag of words" approach
        transforms "bag of words" features using TF-IDF approach
        estimates sentiment using VADER approach
        concatenates sentiment scores to TF-IDF feature matrix
        normalizes data
        chooses best hyper-parameters using grid search
        model choice is based on results of 3-fold cross-validation using squared-hinge loss criterion

    :return: GridSearchCV fit to best model
    '''
    pipeline = Pipeline([
        ('nlp', FeatureUnion([
            ('importance', Pipeline([
                ('bow', CountVectorizer(tokenizer=tokenize, min_df=5, ngram_range=(1, 2))),
                ('tfidf', TfidfTransformer())
            ])),
            ('sentiment', VaderSentiment(tokenize))

        ])),
        ('scale', Normalizer()),
        ('clf', MultiOutputClassifier(LinearSVC(random_state=42, class_weight='balanced'), n_jobs=-1))
    ])

    parameters = {
        'nlp__importance__tfidf__sublinear_tf': [True, False],
        'clf__estimator__C': [0.1, 1, 10, 100, 1000]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=1)

    return cv


def evaluate_model(model, X_test, Y_test, label_names):
    ''' print model evaluation report
        lists F1-score, precision score, and recall score for each label

    :param model: pre-fit sklearn model to evluate
    :param X_test: list of text data from test dataset
    :param Y_test: Pandas DataFrame of labels from test dataset
    :param label_names: list of String label names
    :return: None
    '''
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=label_names))


def save_model(model, model_filepath):
    ''' save serialized model as Python Pickle file

    :param model: pre-fit sklearn model
    :param model_filepath: String filepath to save model
    :return: None
    '''
    with open(model_filepath, 'wb') as file:
        joblib.dump(model.best_estimator_, file)


def main():
    ''' define required command line arguments and goes through model training steps:
        1. load data from SQL database
        2. build and train model
        3. evaluate model and print results
        4. save model to file as serialized Python object

    :return: None
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, label_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, label_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved.')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()