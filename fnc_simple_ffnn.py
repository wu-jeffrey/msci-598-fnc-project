import sys
import numpy as np
import pandas as pd
import os

from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from utils.system import parse_params, check_version

import gensim.downloader as api
from gensim.models import KeyedVectors
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MAX_SEQ_LEN = 4788+40
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 300
BATCH_SIZE=32
N_EPOCHS=10

if __name__ == "__main__":
    check_version()
    parse_params()

    # Load the training dataset and generate folds
    # Note we're only using 1 fold because we're not actually doing k_folds here
    # Just want to leverage as much existing code as possible
    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=1)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    sequences, y = [], []

    print("generating features for training data")
# TOKENIZE TRAINING DATA
    for stance in fold_stances[0]:
        y.append(LABELS.index(stance['Stance']))
        h = stance['Headline']
        b = d.articles[stance['Body ID']]

        sequences.append(h + ' ' + b)

    # tokenize sequences
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(sequences)

    X = tokenizer.texts_to_sequences(sequences)
    X = pad_sequences(X, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
    X = np.array(X)

    print("generating features for test data")
# TOKENIZE TEST DATA
    sequences_test, y_test = [], []

    for stance in hold_out_stances:
        y_test.append(LABELS.index(stance['Stance']))
        h = stance['Headline']
        b = d.articles[stance['Body ID']]

        sequences_test.append(h + ' ' + b)

    # tokenize sequences
    X_test = tokenizer.texts_to_sequences(sequences_test)
    X_test = pad_sequences(X_test, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
    X_test = np.array(X_test)

    print("generating embedding matrix")
    if not os.path.isfile('w2v-google-news-300.kv'):
        embeddings = api.load("word2vec-google-news-300")
        embeddings.save('w2v-google-news-300.kv')
    else:
        embeddings = KeyedVectors.load('w2v-google-news-300.kv')

    # create embeddings matrix
    embeddings_matrix = np.random.uniform(-0.05, 0.05, size=(len(tokenizer.word_index)+1, EMBEDDING_DIM))

    # fill in embeddings from our word2vec model, otherwise leave them randomly initialized
    for word, i in tokenizer.word_index.items(): # i=0 is the embedding for the zero padding
        try:
            embeddings_vector = embeddings[word]
        except KeyError:
            embeddings_vector = None
        if embeddings_vector is not None:
            embeddings_matrix[i] = embeddings_vector

# create neural network
    model = Sequential()

    model.add(Embedding(
        input_dim=(len(embeddings_matrix)),
        output_dim=EMBEDDING_DIM, 
        weights=[embeddings_matrix],
        trainable=False, 
        name="word_embedding_layer",
        input_length=MAX_SEQ_LEN,
        mask_zero=True)
    )

    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dense(4, activation='softmax', name='output_layer'))

    model.summary()

    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(X_test, y_test))

# Run on Holdout set and report the final score on the holdout set
    predicted = [LABELS[np.argmax(a)] for a in model.predict([X_test])]
    actual = [LABELS[int(a)] for a in y_test]

    print("Scores on the dev set")
    report_score(actual,predicted)
    print("")
    print("")

# Fetch and Run on competition test dataset
    competition_dataset = DataSet("competition_test")

    sequences_competition, y_competition = [], []

    for stance in competition_dataset.stances:
        y_competition.append(LABELS.index(stance['Stance']))
        h = stance['Headline']
        b = competition_dataset.articles[stance['Body ID']]

        sequences_competition.append(h + ' ' + b)

    # tokenize sequences
    X_competition = tokenizer.texts_to_sequences(sequences_competition)
    X_competition = pad_sequences(X_competition, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
    X_competition = np.array(X_competition)
    
    # import pdb; pdb.set_trace();
    predicted = [LABELS[np.argmax(a)] for a in model.predict([X_competition])]
    actual = [LABELS[int(a)] for a in y_competition]

    # Gen csv
    answer = {'Headline': [], 'Body ID': [], 'Stance': []}
    for i, stance in enumerate(competition_dataset.stances):
        answer['Headline'].append(stance['Headline'])
        answer['Body ID'].append(stance['Body ID'])
        answer['Stance'].append(predicted[i])

    df = pd.DataFrame(data=answer)

    df.to_csv('answer.csv', index=False, encoding='utf-8')

    print("Scores on the competition test set")
    report_score(actual,predicted)
