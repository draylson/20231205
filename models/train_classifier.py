# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])

# import libraries
import nltk
import numpy as np
import pandas as pd
import pickle
import re
import sqlalchemy as sa
import sys

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier

def load_data(database_filepath):
    '''
    Load movie description and genre data from a database to variables.

    INPUT:
    database_filepath - path to the database file.
    OUTPUT:
    X - independent variables values
    Y - dependent variables values
    category_names - dependent variables labels
    '''
    # load data from database
    engine = sa.create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('GenreByDescription', engine)
    engine.dispose()
    X = df.description.values
    Y = df.iloc[:, 1:].values
    category_names = list(df.columns[1:].values)
    return X, Y, category_names


def tokenize(text):
    '''
    Extract tokens from a text.

    INPUT:
    text - Text from which tokens will be extracted
    OUTPUT:
    clean_tokens - Resulting tokens.
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
    Build a model to predict movie genres.

    OUTPUT:
    model - Resulting model.
    '''
    scorer = make_scorer(recall_score, zero_division=0, average='micro')
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
    ])
    parameters = [
        {
            'clf__estimator': [DecisionTreeClassifier()]
        },
        {
            'clf__estimator': [ExtraTreeClassifier()]
        },
        {
            'clf__estimator': [KNeighborsClassifier()],
            'clf__estimator__n_neighbors': [1, 5, 10]
        },
        {
            'clf__estimator': [RadiusNeighborsClassifier()],
            'clf__estimator__radius': [5.0, 10.0, 20.0]
        },
        {
            'clf__estimator': [ExtraTreesClassifier()],
            'clf__estimator__n_estimators': [50, 100, 200]
        },
        {
            'clf__estimator': [RandomForestClassifier()],
            'clf__estimator__n_estimators': [50, 100, 200]
        }
    ]
    model = GridSearchCV(pipeline, param_grid=parameters, scoring=scorer, cv=3, verbose=4)
    return model


def save_model(model, model_filepath):
    '''
    Save a model to a given file.

    INPUT:
    model - the model to be saved.
    model_filepath - the file in which the model will be saved. 
    '''
    with open(model_filepath,'wb') as f:
        pickle.dump(model, f)


def main():
    '''
    Build and train a model to predict movie genre.
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X, Y)
        print(f"Best params: {model.best_params_}")
        print(f"Best score: {model.best_score_}")

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the IMDB database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \nExample: python '\
              'train_classifier.py ../data/IMDB.db classifier.pkl')


if __name__ == '__main__':
    main()