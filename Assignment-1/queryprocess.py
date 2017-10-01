from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pickle
from math import log10
import heapq
import time
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


def cosine_similarity(term_dictionary, document, query_document):
    similarity = 0
    for term in query_document.term_dict:
        try:
            if document.term_dict[term]:
                similarity += (document.term_dict[term]*query_document.term_dict[term]*term_dictionary[term].idf)
        except KeyError:
            continue
    return similarity


def get_search_results(term_dictionary, document_dictionary, query_tokens, no_of_results):
    similarity_dict = dict()
    query_document = Document(0)
    for query_token in query_tokens:  # Calculate term frequency for query
        if query_token in query_document.term_dict:
            query_document.term_dict[query_token] += 1
        else:
            query_document.term_dict[query_token] = 1
    for query_token in query_document.term_dict:  # Normalize query token
        query_document.term_dict[query_token] = 1 + log10(query_document.term_dict[query_token])
    for document in document_dictionary:
        similarity_dict[document] = cosine_similarity(term_dictionary, document_dictionary[document], query_document)
    heap = [(-value, key) for key, value in similarity_dict.items()]
    largest = heapq.nsmallest(no_of_results, heap)
    largest = [(key, -value) for value, key in largest]
    return largest

lemma_tzr = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
normalized_filename = 'tf-idf-dict-norm-16277.pk'

pun_chars = ['!', '?', ':', '"', ',', '.', '[', ']', '{', '}', '\'', '`', '~', '@', '#', '$', '%', '^', '&', '*', '('
             ')', '+', '=', '|', '\\', '<', '>']
space_chars = ['-', '_']


def tokenize_query(input_query):
    tokens = word_tokenize(input_query)
    final_token = ""
    for token in tokens:
        token = lemma_tzr.lemmatize(token)  # Using WordNet Lemmatization
        token = stemmer.stem(token)  # Using PorterStemmer Method
        if token not in stop_words:
            token = "".join(c for c in token if c not in pun_chars)
            for c in token:
                if c in space_chars:
                    i = token.index(c)
                    token = token[:i] + " " + token[i+1:]
            final_token += (token + " ")
    tokens = word_tokenize(final_token)
    return tokens

with open(normalized_filename, 'rb') as f:
    tf_idf = pickle.load(f)
    term_dict = tf_idf[0]
    document_dict = tf_idf[1]
    print("Loaded Normalized File...")


while True:
    query = input()
    if query == 'q':
        break
    print('Searching in', str(total_no_documents), 'documents')
    start_time = time.time()
    query = tokenize_query(query)
    results = get_search_results(term_dict, document_dict, query, 10)
    print('query took', str(time.time() - start_time), 'seconds')
    print(results)
