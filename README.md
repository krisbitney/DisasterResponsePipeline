# Disaster Response Pipeline Project
This project implements an ETL pipeline, a natural language processing (NLP) pipeline, and a machine learning pipeline. The full pipeline cleans raw text data, applies several transformations to extract and normalize features, trains a Linear Support Vector Machine classifier, and connects the classifier to a web application.

The project uses data from a real disaster relief effort, provided by Figure Eight. There is one raw feature containing text from disaster-related messages, such as requests for medical aid or food. There are 36 message categories that serve as the target variables. Using the messages and category labels, I train a model to classify new disaster relief messages.


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

Pipeline:
1. Extract, Transform, Load (ETL)
2. Natural Language Processing (NLP)
    * Bag of Words
    * TF-IDF
    * Sentiment analysis
3. Normalize data
4. Train Linear Support Vector Classifier (SVC)
    * Select hyperparameters with Grid Search
    * Cross-validation and squared-hinge loss
5. Evaluate performance on test dataset
    * Metrics: precision, recall
6. Import model to web application

The ETL pipeline imports csv data, cleans the data, and stores it in a SQL database. After loading the data from the database, the NLP pipeline transforms the text using a "bag of words" approach that identifies word unigrams and bigrams in the text data to use as features. The "bag of words" features are transformed using a TF-IDF (term frequency - inverse document frequency) method, which adjusts the feature values to account for the frequency of each term's use throughout all of the message data. I also use the VADER sentiment analysis package within NLTK to write a custom transformer that assigns sentiment scores to each message. After concatenating the TF-IDF features and sentiment scores, I normalize the data. I fit a Linear Support Vector Machine classifier using the best hyper-parameters identified with grid search, where "best" is determined by the model's performance in cross-validation with a squared-hinge loss criterion.

I made the web application graphs with Plotly. Udacity provided the remainder of the web application components (i.e. HTML, CSS, Flask implementationm, Bootstrap.js implementation)

The model achieves a weighted average precision score of 0.65, recall score of 0.69, and F1 score of 0.66 across all 36 classes. 

In other words, about 65% of messages belonging to a category are classified as belonging to that category by the model. Similarly, each classified message has about a 69% chance of being a true positive. The model performs much better than this average for common message categories. The model is unreliable for message categories with very few associated messages, and should not be used in those cases.

The model performs well in key areas, including messages related to aid, food, water, shelter, weather, and earthquakes.


### Required libraries

This project uses Python 3 with Numpy, Pandas, Scikit-Learn, NLTK, SQLAlchemy, Plotly, Flask.
