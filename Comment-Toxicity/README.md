# Классификатор токсичности комментариев

Проект представляет собой две обученных модели для задачи классификации 
токсичности комментариев. Обе они различают несколько оттенков токсичности, а именно: 
* ```toxic``` — текст содержит общий токсичный контент. Это может быть грубость, 
оскорбления или другие формы негативного поведения;
* ```severe toxic``` — текст содержит крайне токсичный контент. Эта метка указывает 
на особенно агрессивные или опасные высказывания; 
* ```obscene``` — текст содержит непристойный контент (например, ненормативную 
лексику или сексуально откровенные выражения);
* ```threat``` — текст содержит угрозы. Эта метка указывает на высказывания, которые 
могут быть восприняты как угроза физической расправы или вреда; 
* ```insult``` — текст содержит оскорбления. Это могут быть прямые или косвенные 
высказывания, унижающие или оскорбляющие кого-либо;
* ```identity_hate``` — текст содержит ненависть к определенной группе людей 
(например, расистские, сексистские или гомофобные высказывания).

В качестве обучающей выборки рассматривался крупный датасет для одного из соревнований 
на платформе ```Kaggle``` (прямая ссылка: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/). 
Поскольку датасет масштабный и содержит много информации, была использована его подвыборка в 
$50.000$ первых строк.

## Простой Байесовский классификатор

Первой моделью стал "наивный" классификатор Байеса, делающий предсказания на основе вероятностей 
для слов и последовательностей. Сам он, как и функции векторизации и получения метрик были 
заимствованы из библиотеки ```scikit-learn```. Рассмотрим результат на примере реплики одного из персонажей 
научно-фантастического анимационного сериала «Рик и Морти».

Для входной последовательности ```"Morty, I Need You to Burp SHUT the Hell Up. So shut up or I'll kill you, you little shit!"``` 
имеем

    Prediction for new comment: [('toxic',)]

Модель распознала токсичность, но сделала это в довольно широком и расплывчатом контексте.

## Рекуррентная нейронная сеть

Более проработанной моделью является рекуррентная сеть со следующей архитектурой: 
```Embedding-слой``` с размерностью 32 $\rightarrow$  ```Bidirectional-LSTM-слой``` на 32 нейрона $\rightarrow$ 
```Dense-слой``` на 128 нейронов (```relu```) $\rightarrow$ ```Dense-слой``` на 256 нейронов (```relu```) $\rightarrow$ ```Dense-слой``` 
на 128 нейронов (```relu```) $\rightarrow$ ```Dense-слой``` на 6 нейронов (```sigmoid```).

В качестве функции активации 
использовалась ```BinaryCrossentropy```. Модель прошла обучение в $20$ эпох, что заняло порядка $6$ часов. 
Рассмотрим результат на примере реплики одного из персонажей криминальной драмы «Во все тяжкие».

Для входной последовательности ```"Possum. Big, freaky, lookin’ bitch. Since when did they change it to opossum?"``` 
имеем

    toxic: True
    severe_toxic: False
    obscene: True
    threat: False
    insult: True
    identity_hate: False

Модель распознала токсичность, наличие ненормативной лексики и оскорблений. Как можно видеть, она справляется весьма 
успешно.

## Деплой с помощью Gradio

По окончании тестирования моделей был создан небольшой интерфейс на базе модуля ```Gradio```, использующий рекуррентный 
вариант для предсказаний.

![gradio_demonstration](https://github.com/user-attachments/assets/55edc3f3-5ce2-4ed3-852e-8f8865695619)
