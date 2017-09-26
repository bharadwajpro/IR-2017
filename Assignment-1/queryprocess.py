from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pickle


lemma_tzr = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
normalized_filename = 'tf-idf-norm.pk'

query = ''
pun_chars = ['!', '?', ':', '"', ',', '.', '[', ']', '{', '}', '\'', '`', '~', '@', '#', '$', '%', '^', '&', '*', '('
             ')', '+', '=', '|', '\\', '<', '>']
space_chars = ['-', '_']

tokens = word_tokenize(query)
for token in tokens:
    token = lemma_tzr.lemmatize(token)  # Using WordNet Lemmatization
    token = stemmer.stem(token)  # Using PorterStemmer Method
    if token not in stop_words:
        token = "".join(c for c in token if c not in pun_chars)
        for c in token:
            if c in space_chars:
                i = token.index(c)
                token = token[:i] + " " + token[i+1:]

with open(normalized_filename, 'rb') as f:
    tf_idf = pickle.load(f)
    term_list = tf_idf[0]
    document_list = tf_idf[1]
