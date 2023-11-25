from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    
def describe_cvs():
    data = pd.read_csv('../datasets/labelled_data/all.csv', names=['text', 'entity'], header=None, sep="|")
    print(f"Data loaded into dataframe:\n\n{data.head(10)}\n\n")

    unique_tags = data['entity'].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)
    print(f"Entities in the data:\n\n{unique_tags}\n\n")

    all_tokens = data['entity'].apply(lambda x: len(x.split(" "))).sum(axis = 0)
    print(f"All tokens in ConLL file: {all_tokens}")

    #sent_len = data['text'].apply(len)
    sent_len = data['entity'].apply(lambda x: len(x.split(" ")))
    longest_sentence = sent_len.max()
    shortest_sentence = sent_len.min()
    median_sentence = sent_len.median()
    mean_sentece = sent_len.mean()
    print("Sentence Length Statistics:\n")
    print(f"min: {shortest_sentence}")
    print(f"max: {longest_sentence}")
    print(f"median: {median_sentence}")
    print(f"mean: {mean_sentece}")

    plt.figure(figsize=(10, 6))
    sent_len.plot(kind='hist', bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram of Sentence Lengths')
    plt.xlabel('Length of sentence')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    sns.set(style="whitegrid")

    palette = sns.color_palette("coolwarm", 7)

    plt.figure(figsize=(15, 6))
    sns.boxplot(sent_len, color=palette[3], saturation=0.75, orient="h")

    plt.title('Boxplot of Sentence Lengths', fontsize=16)
    plt.xlabel('Sentence Length', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks([])

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

describe_cvs()
