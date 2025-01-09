import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier

# Загрузка данных
df = pd.read_csv('train.csv')[:50_000]
X = df['comment_text']

# Преобразуем метки в список списков
y = df[df.columns[2:]].apply(lambda row: row[row == 1].index.tolist(), axis=1)

# Преобразуем метки в бинарную матрицу
mlb = MultiLabelBinarizer()
y_binary = mlb.fit_transform(y)

# Проверка форм
print("Shape of X:", X.shape)  # (50000,)
print("Shape of y:", y_binary.shape)  # (50000, 6)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Векторизация текста
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Обучение модели
model = MultiOutputClassifier(MultinomialNB())
model.fit(X_train_vec, y_train)

# Предсказание и оценка модели
y_pred = model.predict(X_test_vec)

# Метрики
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=mlb.classes_))

if __name__ == "__main__":
    comment = "Morty, I Need You to Burp SHUT the Hell Up. So shut up or I'll kill you, you little shit!"
    comment_vec = vectorizer.transform([comment])  # Передаем строку как список
    prediction = model.predict(comment_vec)
    print("\nPrediction for new comment:", mlb.inverse_transform(prediction))