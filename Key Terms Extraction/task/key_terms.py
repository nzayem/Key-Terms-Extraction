import string
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

lem = WordNetLemmatizer()
stop_words = stopwords.words('english')
punc = list(string.punctuation)

file_path = 'news.xml'

with open(file_path, 'r', encoding='utf-8') as file:

    soup = BeautifulSoup(file, 'xml')

    heads = soup.find_all('value', {'name': 'head'})
    contents = soup.find_all('value', {'name': 'text'})

    # Getting the news text:

    news_text = [sorted(word_tokenize(contents[i].text.lower()), reverse=True)for i in range(len(contents))]

    # lemmatization:

    lem_news = [[lem.lemmatize(word) for word in list_words] for list_words in news_text]

    # POS for individual word in each news story:

    pos_news = [[nltk.pos_tag([word])[0] for word in list_words] for list_words in lem_news]

    # Filtering for nouns only:

    nouns_news = [[tup[0] for tup in tup_list if tup[1] == 'NN'] for tup_list in pos_news]

    # Cleaning the nouns list into sentence: Last Stage

    sentence_list = [[' '.join(word)] for word in nouns_news]

    dataset = [sent for element in sentence_list for sent in element]

    # Count the TF-IDF metric for each word in all stories, i.e. apply it to the whole collection of news documents.

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(dataset)

    terms = vectorizer.get_feature_names()

    # The following code was copied from another solution: I have no idea what's going on here, but it gives the result.
    # Again, because there is no topic in the Academy explaining how the get the most frequent words properly based on the score of each term

    for i in range(len(heads)):
        print(f'{heads[i].text}:')
        df = pd.DataFrame(tfidf_matrix[i].T.todense(), index=terms, columns=["tfidf"])
        df.index.name = 'word'
        df = df.sort_values(['tfidf', 'word'], ascending=[False, False]).head(5)
        print(' '.join(list(df.index)), '\n')
