# Подключение библиотек и импортирование нужных функций
import os
import pandas as pd
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from matplotlib import pyplot as plt
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

# Загрузка датасета и отделение фичей от таргета
df = pd.read_csv('train.csv')[:50_000]
X = df['comment_text']
y = df[df.columns[2:]].values

# Векторизация текста
MAX_FEATURES = 200000
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')
vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)

if __name__ == "__main__":
    # Подготовка датасета
    dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
    dataset = dataset.cache()
    dataset = dataset.shuffle(160000)
    dataset = dataset.batch(16)
    dataset = dataset.prefetch(8)

    # Разделение данных на обучение/валидацию/тест
    train = dataset.take(int(len(dataset)*.7))
    val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
    test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))

    # Построение модели
    model = Sequential() 
    model.add(Embedding(MAX_FEATURES+1, 32))
    model.add(Bidirectional(LSTM(32, activation='tanh')))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(6, activation='sigmoid'))
    model.compile(loss='BinaryCrossentropy', optimizer='Adam')

    # Процесс обучения
    history = model.fit(train, epochs=20, validation_data=val)

    # График фукции потерь
    plt.figure(figsize=(8,5))
    pd.DataFrame(history.history).plot()
    plt.show()

    # Получение метрик на тесте
    pre = Precision()
    re = Recall()
    acc = CategoricalAccuracy()

    for batch in test.as_numpy_iterator(): 
        X_true, y_true = batch
        yhat = model.predict(X_true)

        y_true = y_true.flatten()
        yhat = yhat.flatten()

        pre.update_state(y_true, yhat)
        re.update_state(y_true, yhat)
        acc.update_state(y_true, yhat)

    print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')

    # Сохранение обученной модели
    model.save('toxicity.keras')