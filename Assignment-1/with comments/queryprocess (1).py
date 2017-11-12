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





class Term: # general structurog term

    def __init__(self, name): # function to  initialize term

        self.name = name

        self.idf = 0

        self.documents = dict() # dictionary that will store all documents in which term is present



    def update_idf(self):  # function to  update idf of term

        if total_no_documents != 0:

            self.idf = total_no_documents/len(self.documents)





class Document:  # general structurog document

    def __init__(self, doc_id):

        self.id = doc_id

        self.term_dict = dict()





def cosine_similarity(term_dictionary, document, query_document): # function that calculates cosine similarity of every document-vector with query-vector

    similarity = 0

    for term in query_document.term_dict:

        try:

            if document.term_dict[term]:

                similarity += (document.term_dict[term]*query_document.term_dict[term]*term_dictionary[term].idf) # performing dot product of vectors

        except KeyError:

            continue

    return similarity





def get_search_results(term_dictionary, document_dictionary, query_tokens, no_of_results): # function that returns 10 documents with best results of query

    similarity_dict = dict()

    query_document = Document(0)

    for query_token in query_tokens:  # Calculate term frequency for query

        if query_token in query_document.term_dict:

            query_document.term_dict[query_token] += 1

        else:

            query_document.term_dict[query_token] = 1

    for query_token in query_document.term_dict:  # Normalize query token

        query_document.term_dict[query_token] = 1 + log10(query_document.term_dict[query_token])

    for document in document_dictionary: # calculating cosine similarity of every document-vector with query-vector

        similarity_dict[document] = cosine_similarity(term_dictionary, document_dictionary[document], query_document)

    heap = [(-value, key) for key, value in similarity_dict.items()] # store all document ids with their cosine similarity in max heap

    largest = heapq.nsmallest(no_of_results, heap)  # will return 10 documents with hghest similarity

    largest = [(key, -value) for value, key in largest]  

    return largest



lemma_tzr = WordNetLemmatizer()

stemmer = PorterStemmer()

stop_words = set(stopwords.words('english'))

normalized_filename = 'tf-idf-dict-norm-16277.pk' # file that has tokenized term-list and document-list tf-idf structures



pun_chars = ['!', '?', ':', '"', ',', '.', '[', ']', '{', '}', '\'', '`', '~', '@', '#', '$', '%', '^', '&', '*', '('

             ')', '+', '=', '|', '\\', '<', '>']

space_chars = ['-', '_']





def tokenize_query(input_query): # function that processes query, lemmatizing it, stemming it and removing stop words from it

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



with open(normalized_filename, 'rb') as f: # get document-list and term-list datastructure we created while calculating tf-idf

    tf_idf = pickle.load(f)

    term_dict = tf_idf[0]

    document_dict = tf_idf[1]

    print("Loaded Normalized File...")





while True:

    query = input()  # input query

    if query == 'q':  

        break

    print('Searching in', str(total_no_documents), 'documents')  # printing total no of document we are searching for

    start_time = time.time()  # store time before start of searching documents

    query = tokenize_query(query) # preprocess query

    results = get_search_results(term_dict, document_dict, query, 10) # function that returns top 10documents with highestsimilarity as results

    print('query took', str(time.time() - start_time), 'seconds')  # time taken to print results

    print(results)