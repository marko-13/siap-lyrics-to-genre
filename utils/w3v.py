import os

import nltk
import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")
porter = PorterStemmer()


def preprocess(train_captions):
    new_captions = []
    for caption in train_captions:
        new_captions.append(
            " ".join(
                [
                    porter.stem(word)
                    for word in caption
                    if word not in stopwords.words("english")
                ]
            )
        )
    return new_captions


def data_na_value_cleaning(data_loc):
    data_loc.dropna(inplace=True)
    data_loc.reset_index(inplace=True, drop=True)

    return data_loc


lyrics = []
for csv in os.listdir("../data/lyrics_csv"):
    data = pd.read_csv("../data/lyrics_csv/" + csv, delimiter=";")
    data = data_na_value_cleaning(data)
    lyrics.extend(list(data["lyrics"]))

data = [x.lower().split() for x in lyrics]
data = preprocess(data)
print("Started")
model = Word2Vec(sentences=data, size=200, window=4, min_count=1, workers=4)
model.save("../models/word2vec.model")
# model = Word2Vec.load("word2vec.model")
# print(model.wv["mafia"])
