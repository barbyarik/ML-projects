'''A file with a naive approach to the task using a sentiment dictionary'''

import matplotlib.pyplot as plt
import string
from collections import Counter

naive_stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                    "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
                    "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
                    "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
                    "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
                    "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
                    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
                    "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
                    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
                    "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]   

def naive_crearing_tokenization(uncleaned_text):
    '''Easy text cleaning and space separation'''

    lower_text = uncleaned_text.lower()
    cleaned_text = lower_text.translate(str.maketrans('', '', string.punctuation))
    tokenized_text = cleaned_text.split()
    final_text = []
    for word in tokenized_text:
        if word not in naive_stop_words:
            final_text.append(word)

    return final_text

def naive_analysis(processed_text, path='data/textfiles/'):
    '''Detecting emotions in a dictionary-based text'''

    emotion_list = []
    with open(f'{path}emotions.txt', 'r') as file:
        for line in file:
            clear_line = line.replace("\n", '').replace(",", '').replace("'", '').replace(" ", '').strip()
            word, emotion = clear_line.split(':')
            if word in processed_text:
                emotion_list += [emotion] * processed_text.count(word)

    return Counter(emotion_list)

def naive_emotion_plot(counter_obj, save_name='naive_emotion.png', path='data/plots/'):
    fig, ax1 = plt.subplots()
    ax1.bar(counter_obj.keys(), counter_obj.values())
    fig.autofmt_xdate()
    ax1.set_title('Naive sentiment analysis')
    ax1.set_xlabel('emotions')
    ax1.set_ylabel('count')
    plt.savefig(f'{path}{save_name}')
    plt.show()

if __name__ == '__main__':
    from support import read_from_file
    text = read_from_file()
    processed_text = naive_crearing_tokenization(text)
    counter = naive_analysis(processed_text)
    print(counter)
    naive_emotion_plot(counter, 'naive_emotion.png')
