import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import re  # za regex izraze, da se nadju samo reci
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#importing the glove library
from glove import Corpus, Glove
from mittens import GloVe
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer

MAX_NUM_WORDS = 1000
MAX_SEQUENCE_LENGTH = 100


def main_embedding():
    nltk.download('stopwords')
    stop = set(stopwords.words('english'))
    lyrics_data = []
    y_label = []
    lyrics_data, y_label = create_data_input(lyrics_data, y_label)
    preprocessed_words = word_preprocessing(lyrics_data, stop)
    glove_input = lemmatization(preprocessed_words)

    print(glove_input)

    get_embeddings(glove_input)


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


'''def tokenize(preprocessed):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index    

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(train['target']))
    print(data.shape)
    print(labels.shape)'''


def get_embeddings(prepared_input):
    corpus = Corpus()
    corpus.fit(prepared_input, window=10)
    glove = Glove(no_components=5, learning_rate=0.05)
    glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    glove.save('glove.model')


main_embedding()
