from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pickle
total_no_documents = 16277
docs_found = dict()


class Term:
    def __init__(self, name):
        self.name = name
        self.idf = 0
        self.documents = dict()

    def update_idf(self):
        if total_no_documents != 0:
            self.idf = total_no_documents/len(self.documents)


class Document:
    def __init__(self, doc_id):
        self.id = doc_id
        self.term_dict = dict()

lemma_tzr = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
normalized_filename = 'tf-idf-dict-norm-16277.pk'

query = 'Nothing you can sell me, I\'ll see you around'
pun_chars = ['!', '?', ':', '"', ',', '.', '[', ']', '{', '}', '\'', '`', '~', '@', '#', '$', '%', '^', '&', '*', '('
             ')', '+', '=', '|', '\\', '<', '>']
space_chars = ['-', '_']

tokens = word_tokenize(query)
print(tokens)
for token in tokens:
    token = lemma_tzr.lemmatize(token)  # Using WordNet Lemmatization
    token = stemmer.stem(token)  # Using PorterStemmer Method
    if token not in stop_words:
        token = "".join(c for c in token if c not in pun_chars)
        for c in token:
            if c in space_chars:
                i = token.index(c)
                token = token[:i] + " " + token[i+1:]
                tokens.
exit()

with open(normalized_filename, 'rb') as f:
    tf_idf = pickle.load(f)
    term_dict = tf_idf[0]
    document_dict = tf_idf[1]


for token in tokens:
    for doc in document_dict:
        if token in document_dict[doc].term_dict and doc in docs_found:
            docs_found[doc] = 1

for docs in docs_found:
    if docs_found[docs] == 4:
        print(docs, docs_found[docs])
print(len(tokens), tokens)
