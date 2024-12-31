import numpy as np
import random

from train import SEQ_LENGTH, characters, char_to_index, index_to_char

def _sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(model, text, length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    for _ in range(length):
        X  = np.zeros((1, SEQ_LENGTH, len(characters)), dtype=np.bool_)
        for t, character in enumerate(sentence):
            X[0, t, char_to_index[character]] = 1

        predictions = model.predict(X, verbose=0)[0]
        next_index = _sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated