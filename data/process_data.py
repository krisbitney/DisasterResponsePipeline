import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    ''' load data from csv

    :param messages_filepath: String filepath of text data
    :param categories_filepath: String filepath of targets/labels
    :return: merged Pandas DataFrame containing text data and targets/labels
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, on='id')


def clean_data(df):
    ''' ETL pipeline

    :param df: Pandas DataFrame containing text data and targets/labels
    :return: Pandas DataFrame containing cleaned data
    '''
    categories = df['categories'].str.split(';', expand=True)

    # set column names to category names
    row = categories.loc[0, :]
    category_colnames = list(row.apply(lambda x: x[0:-2]))
    categories.columns = category_colnames

    # convert categories to indicator variables, ensure columns have two classes, drop non-binary values
    drop_idx = []
    for column in categories:
        categories[column] = categories[column].astype(str).apply(lambda x: x[-1])
        categories[column] = categories[column].astype(int)
        # drop if less than two classes
        if len(categories[column].unique()) < 2:
            categories = categories.drop(columns=column)
            continue
        # ensure values are either 0 or 1 in every column
        drop_idx += categories[(categories[column] != 0) & (categories[column] != 1)].index.tolist()
    categories = categories.drop(index=drop_idx)

    # replace messy categories column with cleaned categories columns
    df = df.drop(columns=['categories'])
    df = df.merge(categories, left_index=True, right_index=True)


    # drop true duplicates
    df = df.drop_duplicates()
    # drop duplicates in terms of message text
    df = df.sample(frac=1, random_state=42)
    df = df.drop_duplicates(subset='message')

    return df


def save_data(df, database_filename):
    ''' save cleaned data to Pandas DataFrame

    :param df: Pandas DataFrame to save to database
    :param database_filename: String file name for SQL database
    :return: None
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('etl', engine, index=False, if_exists='replace')


def main():
    ''' define required command line arguments and go through ETL steps:
        1. extract data from csv files
        2. clean/transform data
        3. save data to SQL database

    :return: None
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database.')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument, respectively, and '\
              'the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()