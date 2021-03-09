import pandas as pd


# RUN THIS FILE ONLY ONCE, WHEN YOU ARE CREATING DATASET WITH BASIC 10 MUSIC GENRES
def crop_dataset():
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1000)

    df = pd.read_csv('../data/music_genres/all_genres.csv', delimiter=';')
    df = df.loc[df['genre_name'].isin(['pop', 'rock', 'blues', 'classical', 'country',
                                      'disco', 'hip hop', 'jazz', 'metal', 'reggae'])]

    df.to_csv('basic_genres.csv', sep=';', encoding='utf-8')


crop_dataset()
