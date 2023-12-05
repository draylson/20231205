# import libraries
import pandas as pd
import sqlalchemy as sa
import sys

def clean_data(df):
    '''
    Clean a pandas dataframe.

    INPUT:
    df - original pandas dataframe
    OUTPUT:
    df - cleaned pandas dataframe
    '''
    df = pd.DataFrame(df[['description', 'genre']])
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df['genre'] = df['genre'].apply(lambda x: x.translate(str.maketrans({"[": "", "]": "", "'": '', " ": ''})))
    dummies = df['genre'].str.get_dummies(sep=',')
    df = df.join(dummies)
    df.drop('genre', axis=1, errors='ignore', inplace=True)
    return df


def save_data(df, database_filename):
    '''
    Save a pandas dataframe into a file.

    INPUT:
    df - pandas dataframe
    database_filename - file in which the dataframe will be saved 
    '''
    engine = sa.create_engine('sqlite:///' + database_filename)
    df.to_sql('GenreByDescription', engine, index=False, if_exists='replace')  


def main():
    '''
    Process disaster messages data and save them into a database file.
    '''
    if len(sys.argv) == 3:

        csv_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    CSV FILE: {}'.format(csv_filepath))
        df = pd.read_csv(csv_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepath of the IMDB datasets as the first '\
              'argument, as well as the filepath of the database to save the '\
              'cleaned data to as the second argument. \nExample: python '\
              'process_data.py imbd_raw.csv IMDB.db')


if __name__ == '__main__':
    main()