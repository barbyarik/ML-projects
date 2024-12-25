'''Using the NLTK library to tokenize and predict the sentiment of a text'''

import string
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

from sentiment_naive import naive_analysis, naive_emotion_plot

def nltk_crearing_tokenization(uncleaned_text):
    '''Cleaning and tokenization using VADER'''

    lower_text = uncleaned_text.lower()
    cleaned_text = lower_text.translate(str.maketrans('', '', string.punctuation))
    tokenized_text = word_tokenize(cleaned_text, "english")
    final_text = []
    for word in tokenized_text:
        if word not in stopwords.words('english'):
            final_text.append(word)
    lemma_words = []
    for word in final_text:
        word = WordNetLemmatizer().lemmatize(word)
        lemma_words.append(word)

    return lemma_words

def nltk_analysis(sentiment_text):
    '''NLTK Prediction'''

    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    if max(score.values()) == score['neg']:
        ans = "Negative Sentiment"
    elif max(score.values()) == score['pos']:
        ans = "Positive Sentiment"
    else:
        ans = "Neutral Sentiment"
    
    return (ans, score)


if __name__ == '__main__':
    from support import read_from_file
    text = read_from_file()
    processed_text = nltk_crearing_tokenization(text)
    counter = naive_analysis(processed_text)
    print(counter)
    lower_text = text.lower()
    cleaned_text = lower_text.translate(str.maketrans('', '', string.punctuation))
    print(nltk_analysis(cleaned_text))
    naive_emotion_plot(counter, 'naive_emotion.png')