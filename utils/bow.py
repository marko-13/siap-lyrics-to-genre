from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score

def get_bow():
    nltk.download('punkt')
    nltk.download('stopwords')

    # U corpus listi ce se nalaziti 10 stringova, svaki string je skup svih lyricsa tog zanra
    corpus = []
    corpus_temp = []
    # U tokenized corpus listi ce se nalaziti liste svih reci iz lyricsa jednog zanra
    tokenized_corpus = []

    y_label = []
    temp_genre = ['blues', 'classical', 'country', 'disco', 'hip hop', 'jazz', 'metal',
                  'pop', 'reggae', 'rock']

    for genre in temp_genre:
        df = pd.read_csv('../data/lyrics_csv/' + genre + '.csv', delimiter=';', encoding='utf-8')

        for index, row in df.iterrows():
            corpus_temp.append(re.sub('[^A-Za-z ]', ' ', row['lyrics']).lower())
            y_label.append(genre)

        # corpus.append(corpus_temp)
        # tokenized_corpus.append(word_tokenize(corpus_temp))
        # corpus_temp = []

    # for tokenized_corpus_temp in tokenized_corpus:
    #     for word in tokenized_corpus_temp:
    #         if word in stopwords.words('english'):
    #             tokenized_corpus_temp.remove(word)
    stemmer = PorterStemmer()

    matrix = CountVectorizer()
    X = matrix.fit_transform(corpus_temp).toarray()
    print(X.shape)

    x_train, x_test, y_train, y_test = train_test_split(X, y_label)

    classifier = GaussianNB()
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)


get_bow()
