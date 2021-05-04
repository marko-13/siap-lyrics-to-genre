import sys

import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import re  # za regex izraze, da se nadju samo reci
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer
import os

from keras.layers import Embedding
from keras.initializers import Constant

MAX_NUM_WORDS = 1000
MAX_SEQUENCE_LENGTH = 100   # probaj i sa 50

nltk.download('stopwords')
stop = set(stopwords.words('english'))

'''def main_embedding():
    embeddings_index = glove_dataset()

    nltk.download('stopwords')
    stop = set(stopwords.words('english'))
    lyrics_data = []
    y_label = []
    lyrics_data, y_label = create_data_input(lyrics_data, y_label)
    preprocessed_words = word_preprocessing(lyrics_data, stop)
    data, labels, word_index = tokenize(preprocessed_words, y_label)
    get_embeddings(word_index, embeddings_index)'''


# ucitava stanfordov glove model i pravi od njega recnik, ovo ce potrajati koji minut
# kljuc je rec a vrednost je vektor
# TO DO: preuzmi glove.txt i promeni putanje na liniji 43
def glove_dataset():
    embeddings_index = {}
    fpath = os.path.join('C:/Users/Ivana/Desktop/FTN/MAS/SIAP/glove.42B.300d', 'glove.42B.300d.txt')
    args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
    with open(fpath, **args) as f:
        for line in f:
            print("ucitava linije...")
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))     # 1917494 vektora
    f.close()

    return embeddings_index


def create_data_input(lyrics_data, y_label):
    temp_genre = ['blues', 'classical', 'country', 'disco', 'hip hop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    for genre in temp_genre:
        df = pd.read_csv('../data/lyrics_csv/' + genre + '.csv', delimiter=';', encoding='utf-8')

        for index, row in df.iterrows():
            lyrics_data.append(row['lyrics'])
            y_label.append(genre)

    return lyrics_data, y_label


def word_preprocessing(lyrics_data, stop):
    preprocessed_lyrics = []
    for lyrics in lyrics_data:
        lyrics.lower()
        words_only = re.findall(r'(?:\w+)', lyrics, flags=re.UNICODE)  # ukloni interpunkciju i sl
        line = []  # temp da tu cuvam sve reci jednog liricsa
        for word in words_only:  # remove stop words
            if word not in stop:
                line.append(word)
        # line.strip()
        preprocessed_lyrics.append(line)

    return preprocessed_lyrics


def lemmatization(preprocessed):
    nltk.download('wordnet')
    wordnet_lemmatizer = WordNetLemmatizer()
    preprocessed_with_lemmas = []
    for lyrics_words in preprocessed:
        np_lyrics = np.array(lyrics_words)
        for word in lyrics_words:
            np_lyrics = np.where(np_lyrics == word, wordnet_lemmatizer.lemmatize(word), np_lyrics)
        preprocessed_with_lemmas.append(np_lyrics)

    return preprocessed_with_lemmas    # niz nizova reci ali lematizovanih


def tokenize(preprocessed, y_labels):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(preprocessed)
    sequences = tokenizer.texts_to_sequences(preprocessed)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(y_labels))
    #labels = np.asarray(y_labels).reshape
    print(data.shape)       # train podaci
    print(labels.shape)     # labele tj zanrovi

    return data, labels, word_index


# embedding index je ucitan iz glove.txt i pretvoren u recnik
def get_embeddings(word_index, embeddings_index):
    EMBEDDING_DIM = 300
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:    # proverava da li se rec nalazi u glove recniku
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                weights=[embedding_matrix],
                                trainable=True)        # moze i true

    '''embedding_layer = Embedding(vocabulary_size,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=True)'''

    return embedding_layer


# mapira svaki zanr u niz od 10 kolona, 1 je tamo gde je lyrics, helper metoda zbog to_categorical
# na primer (za to_categorical) [ 1 0 0 0 0 0 0 0 0 0 ] za blues
def labels_to_num(y_labels):
    label_nums = []
    for label in y_labels:
        if label == 'blues':
            label_nums.append(0)
        elif label == 'classical':
            label_nums.append(1)
        elif label == 'country':
            label_nums.append(2)
        elif label == 'disco':
            label_nums.append(3)
        elif label == 'hip hop':
            label_nums.append(4)
        elif label == 'jazz':
            label_nums.append(5)
        elif label == 'metal':
            label_nums.append(6)
        elif label == 'pop':
            label_nums.append(7)
        elif label == 'reggae':
            label_nums.append(8)
        elif label == 'rock':
            label_nums.append(9)

    return label_nums


# za proslu kt
def test_clyrics(input_test, output, model):
    loss, accuracy = model.evaluate(input_test, output, verbose=False)
    accuracy = round(accuracy * 100) - loss
    print("Test Accuracy: ", accuracy, "%")






