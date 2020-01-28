import sys
import pandas as pd
import re,string,unicodedata
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


# Basic Functions
def remove_tags(text):
    """ Removes html tags """

    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

def remove_brackets(text):
    """Removes brackets"""
    return re.sub('\[[^]]*\}', '', text)

def remove_all_noise(text):
    text = remove_tags(text)
    text = remove_brackets(text)
    return text

def remove_sp_char(text, remove_digits=True):
    """ Strips text of special char """
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def stemmer(text):
    porter = PorterStemmer()
    text = ' '.join([porter.stem(word) for word in text.split()])
    return text

def remove_stopwords(text, is_lowercase = False):
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    unwanted = stopwords.words('english')

    if is_lowercase:
        filtered_tokens = [token for token in tokens if token not in unwanted]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in unwanted]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


# import data from csv file
movie_df = pd.read_csv('dataset.csv') # movie reviews df

movie_df['review'] = movie_df['review'].apply(remove_all_noise) # apply only on review column
movie_df['review'] = movie_df['review'].apply(remove_sp_char)
movie_df['review'] = movie_df['review'].apply(remove_stopwords)
movie_df['review'] = movie_df['review'].apply(stemmer) 

# print(movie_df)

movie_df.to_csv('filtered_dataset.csv')
