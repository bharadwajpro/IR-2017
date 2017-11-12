from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from gensim import corpora
from gensim.summarization.bm25 import BM25
import time
from tqdm import trange
import os
import pickle
import heapq

total_no_documents = 16277
processed_songs_folder = './processed_songs/'
files = os.listdir(processed_songs_folder)
no_of_files_to_process = 500
corpus_filename = 'bm25-corpus-' + str(no_of_files_to_process) + '.pk'
texts_filename = 'bm25-texts-' + str(no_of_files_to_process) + '.pk'
docs_name_filename = 'bm25-docs-' + str(no_of_files_to_process) + '.pk'
save_for_every = 100
lemma_tzr = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
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
                    j = token.index(c)
                    token = token[:j] + " " + token[j+1:]
            final_token += (token + " ")
    tokens = word_tokenize(final_token)
    return tokens

try:
    with open(corpus_filename, 'rb') as fp:
        corpus = pickle.load(fp)
        i = len(corpus)
    with open(texts_filename, 'rb') as fp:
        texts = pickle.load(fp)
    with open(docs_name_filename, 'rb') as fp:
        documents = pickle.load(fp)
except FileNotFoundError:
    i = 0
    corpus = []
    texts = []
    documents = []

for i in trange(i, len(files[:no_of_files_to_process])):
    filename = files[i]
    # if i <= 5:
    #     print(filename)
    if i % save_for_every == 0 and i >= 100:
        output = open(corpus_filename, 'wb')
        pickle.dump(texts, output)
        output.close()
        output = open(texts_filename, 'wb')
        pickle.dump(texts, output)
        output.close()
        output = open(docs_name_filename, 'wb')
        pickle.dump(documents, output)
        output.close()
        print('Saved corpus to file', corpus_filename, 'Saved texts, documents to files', texts_filename, docs_name_filename)
    with open(processed_songs_folder + filename, 'r') as f:
        documents.append(filename[:-4])
        content = f.read()
        words = word_tokenize(content)
        texts.append(words)
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

    if i == len(files[:no_of_files_to_process]) - 1:
        output = open(corpus_filename, 'wb')
        pickle.dump(texts, output)
        output.close()
        output = open(texts_filename, 'wb')
        pickle.dump(texts, output)
        output.close()
        output = open(docs_name_filename, 'wb')
        pickle.dump(documents, output)
        output.close()
        print('Saved corpus to file', corpus_filename, 'Saved texts, documents to files', texts_filename, docs_name_filename)

bm25_obj = BM25(corpus)
while True:
    scores_dict = dict()
    query = input()
    if query == 'q':
        break
    print('Searching in', str(no_of_files_to_process), 'documents')
    start_time = time.time()
    query = tokenize_query(query)
    average_idf = sum(map(lambda k: float(bm25_obj.idf[k]), bm25_obj.idf.keys())) / len(bm25_obj.idf.keys())
    scores = bm25_obj.get_scores(query, average_idf)
    for i in range(len(files[:no_of_files_to_process])):
        scores_dict[files[i]] = scores[i]
    heap = [(-value, key) for key, value in scores_dict.items()]
    largest = heapq.nsmallest(10, heap)
    results = [(key, -value) for value, key in largest]
    print('query took', str(time.time() - start_time), 'seconds')
    print(results)
