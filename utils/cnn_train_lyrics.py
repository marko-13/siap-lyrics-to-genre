from keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout, concatenate, Conv1D, MaxPooling1D, \
    GlobalMaxPooling1D
from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.models import Model
from keras import regularizers


from utils.glv import *

from sklearn.model_selection import train_test_split
from sklearn import metrics

stop = set(stopwords.words('english'))


def main():
    lyrics_data = []
    y_label = []
    lyrics_data, y_label = create_data_input(lyrics_data, y_label)

    # podeli na train 80% test 10% validation 10%
    X_train, X_test, y_train, y_test = train_test_split(lyrics_data, y_label, test_size=0.2, random_state=123)
    #X_test, x_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=123)

    # ucitaj glove
    embeddings_index = glove_dataset()

    # sredjivanje train skupa
    preprocessed_words = word_preprocessing(X_train, stop)
    data, labels, word_index = tokenize(preprocessed_words, labels_to_num(y_train))

    # sredjivanje validacionog
    #preprocessed_words_val = word_preprocessing(x_val, stop)
    #data_val, labels_val, word_index1 = tokenize(preprocessed_words_val, labels_to_num(y_val))
    preprocessed_words = word_preprocessing(X_test, stop)
    data_val, labels_val, word_index = tokenize(preprocessed_words, labels_to_num(y_test))

    embedding_layer = get_embeddings(word_index, embeddings_index)  # ovo ce biti ulaz za cnn

    # train CNN
    model = init_cnn(data, labels, embedding_layer, embeddings_index, data_val, labels_val)

    # test
    test_clyrics(data, labels, model)


# formiranje konvolucione mreze, input je embedding layer formiran na osnovu glove embeddinga
# konv primenjuje 128 filtera na 3 reci, izmedju se ubacuje dropout sloj zbog overfitinga
# early stopping - treniraj dok se ne smanji validation loss, nije dalo sjajne rezultate
def init_cnn(X_train, labels, embedding_layer, embeddings_index, x_val, y_val):
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 3, activation='relu')(embedded_sequences)
    x = Dropout(0.2)(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(10, activation='relu')(x)
    # ok je i softmax aktivaciona sa rmsprop optimizerom, oko 75%
    preds = Dense(10, activation='sigmoid')(x)  # za 10 klasa zanrova, 10 izlaznih predikcija,

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',   # probala sam i binary_crossentropy, ovaj je bolji
                  optimizer='adam',          # od optimizera rmsprop, adam
                  metrics=['acc'])

    history = model.fit(X_train, labels,
                        batch_size=128,
                        epochs=20,
                        validation_data=(x_val, y_val))

    loss, accuracy = model.evaluate(X_train, labels, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))

    return model


def test_lyrics(x_test, y_test, model):
    preprocessed_words = word_preprocessing(x_test, stop)
    data, labels, word_index = tokenize(preprocessed_words, labels_to_num(y_test))

    loss, accuracy = model.evaluate(data, labels, verbose=False)
    print("Test Accuracy: {:.2f}".format(accuracy), "%")

    pred_test = model.predict(data)
    #print("TESTED")


main()