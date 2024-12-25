'''Using the TextBlob library to tokenize and predict the sentiment of a text'''

from textblob import Word, TextBlob
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = stopwords.words('english')

def preprocess_text(uncleaned_text):
    '''Text preprocessing and cleaning'''

    processed_text = uncleaned_text
    processed_text = re.sub(r'[^\w\s]', '', processed_text)
    processed_text = " ".join(word for word in processed_text.split() if word not in stop_words)
    processed_text = " ".join(Word(word).lemmatize() for word in processed_text.split())
    return processed_text

def text_blob_analysis(processed_text):
    '''Predicting the sentiment of a text'''

    return TextBlob(processed_text).sentiment

if __name__ == '__main__':
    from support import read_from_file
    text = read_from_file()
    processed_text = preprocess_text(text)
    print(text_blob_analysis(processed_text))