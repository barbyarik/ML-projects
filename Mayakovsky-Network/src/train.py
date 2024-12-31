import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

filepath = os.path.join('data', 'Mayakovsky.txt')  # for root launch

text = open(filepath, 'r', encoding='utf-8').read().lower()

characters = sorted(set(text))

char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40
STEP_SIZE  = 3

sentences = []
next_characters = []

if __name__ == "__main__":
    for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
        sentences.append(text[i: i+SEQ_LENGTH])
        next_characters.append(text[i+SEQ_LENGTH])

    X = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool_)
    y = np.zeros((len(sentences), len(characters)), dtype=np.bool_)

    for i, sentence in enumerate(sentences):
        for t, character in enumerate(sentence):
            X[i, t, char_to_index[character]] = 1
        y[i, char_to_index[next_characters[i]]] = 1

    model = Sequential()
    model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters)), return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(len(characters)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

    model.fit(X, y, batch_size=256, epochs=100)

    model.save('mayakovsky_trained.keras')