from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
import nltk


def get_corpus():
    corpus = []

    y_label = []
    temp_genre = ['blues', 'classical', 'country', 'disco', 'hip hop', 'jazz', 'metal',
                  'pop', 'reggae', 'rock']

    for genre in temp_genre:
        df = pd.read_csv('../data/lyrics_csv/' + genre + '.csv', delimiter=';', encoding='utf-8')

        for index, row in df.iterrows():
            corpus.append(row['lyrics'])
            y_label.append(genre)

    count_vect = CountVectorizer()
    x_train_counts = count_vect.fit_transform(corpus)
    print(x_train_counts.shape)


def get_bow():
    # count_vec = CountVectorizer()
    # x_train_counts = count_vec.fit_transform('../data/lyrics/blues.txt')
    # print(x_train_counts.shape)
    temp_genre = 'rock'
    df = pd.read_csv('../data/lyrics_csv/' + temp_genre + '.csv', delimiter=';')


get_corpus()
