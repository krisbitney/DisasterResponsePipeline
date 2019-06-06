# Disaster Response Pipeline Project
This project implements an ETL pipeline, a natural language processing (NLP) pipeline, and a machine learning pipeline. The full pipeline cleans raw text data, applies several transformations to extract and normalize features, trains a Linear Support Vector Machine classifier, and connects the classifier to a web application.

The project uses data from a disaster relief effort. There is one raw feature containing text from disaster-related messages, such as requests for medical aid or food. There are 36 message categories that serve as the target variables. Using the messages and category labels, I train a model to classify new disaster relief messages.


### Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Methodology

The ETL pipeline imports csv data, cleans the data, and stores it in a SQL database. After loading the data from the database, the NLP pipeline transforms the text using a "bag of words" approach that identifies word unigrams and bigrams in the text data to use as features. The "bag of words" features are transformed using a TF-IDF (term frequency - inverse document frequency) method, which adjusts the feature values to account for the frequency of term use throughout all of the message data. I also use the VADER sentiment analysis method to assign sentiment scores to each message. After concatenating the TF-IDF features and sentiment scores, I normalize the data. A Linear Support Vector Machine classifier is produced using the best hyper-parameters identified with grid search, where "best" is determined by the model's performance in cross-validation with a squared-hinge loss criterion.

The model achieves an average precision score of 0.54, recall score of 0.65, and F1 score of 0.59 across all 36 classes. 

In other words, about 54% of messages belonging to a category are classified as belonging to that category by the model. Similarly, each classified message has about a 65% chance of being a true positive.
