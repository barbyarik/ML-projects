'''Sentiment prediction based on the Roberta transformer model from HuggingFace'''

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

roberta = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(roberta)
model = AutoModelForSequenceClassification.from_pretrained(roberta)
labels = ['Negative', 'Neutral', 'Positive']

def roberta_crearing_tokenization(uncleaned_text):
    '''Cleaning and tokenization'''

    text_words = []
    for word in uncleaned_text.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        
        elif word.startswith('http'):
            word = "http"
        text_words.append(word)

    text_proc = " ".join(text_words)
    encoded_text = tokenizer(text_proc, return_tensors='pt')

    return encoded_text

def roberta_analysis(encoded_text):
    '''Roberta Prediction'''

    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    for i in range(len(scores)):
        l = labels[i]
        s = scores[i]
        print(l,s)

if __name__ == '__main__':
    from support import read_from_file
    text = read_from_file()[:512]
    encoded_text = roberta_crearing_tokenization(text)
    roberta_analysis(encoded_text)