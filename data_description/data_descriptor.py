from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_data(text, remove_stopwords = True):
    nltk.download('punkt')
    nltk.download('stopwords')

    tokens = word_tokenize(text)

    if remove_stopwords:
        tokens = [word.lower() for word in tokens if word.isalpha()]
        tokens = [word for word in tokens if word not in stopwords.words('english')]

    return tokens

def generate_wordcloud(tokens):
    wordcloud = WordCloud(width = 800, height = 800,
                    background_color ='white',
                    stopwords = stopwords.words('english'),
                    min_font_size = 10).generate(' '.join(tokens))

    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()

def generate_most_common(tokens):
    freq_dist = Counter(tokens)
    most_common = freq_dist.most_common(20)

    plt.figure(figsize=(12, 8))

    sns.barplot(x=[val[1] for val in most_common], y=[val[0] for val in most_common])
    plt.show()

with open('entities.txt', 'r') as file:
    entities_tokens = file.readlines() #no preprocessing needed

with open('text.txt', 'r') as file:
    text = file.read()

text_tokens = preprocess_data(text)

generate_wordcloud(text_tokens)
generate_most_common(text_tokens)

generate_wordcloud(entities_tokens)
generate_most_common(entities_tokens)
