import os
from tqdm import trange
from nltk import word_tokenize
from math import log10, sqrt
import pickle

processed_songs_folder = './processed_songs/'
files = os.listdir(processed_songs_folder)
no_of_files_to_process = 5000
output_filename = 'tf-idf-' + str(no_of_files_to_process) + '.pk'
normalized_filename = 'tf-idf-norm.pk'
total_no_documents = 0


class TfIdfList(list):
    def __init__(self, *args):
        list.__init__(self, *args)

    def __contains__(self, item):
        for pos in range(len(self)):
            if item == str(self[pos]):
                return True
        return False

    def index(self, value, start=None, stop=None):
        pos = 0
        while pos < len(self):
            if str(self[pos]) == value:
                break
            else:
                pos += 1
        return pos


class Term:
    def __init__(self, name):
        self.name = name
        self.idf = 0
        self.documents = set()

    def __str__(self):
        return self.name

    def update_idf(self):
        if total_no_documents != 0:
            self.idf = total_no_documents/len(self.documents)


class Document:
    def __init__(self, doc_id):
        self.id = doc_id
        self.term_dict = dict()

    def __str__(self):
        return self.id


try:
    with open(output_filename, 'rb') as fp:
        temp_list = pickle.load(fp)
        term_list = temp_list[0]
        document_list = temp_list[1]
        i = len(document_list)
except FileNotFoundError:
    term_list = TfIdfList()
    document_list = TfIdfList()
    i = 0


def construct_tf_idf(tokens, doc_id):
    global total_no_documents, term_list, document_list
    total_no_documents += 1
    for term in term_list:
        term.update_idf()
    for token in tokens:
        if token not in term_list:
            term_list.append(Term(token))

        index = term_list.index(token)

        term_list[index].documents.add(doc_id)
        term_list[index].update_idf()

        if doc_id in document_list:
            index = document_list.index(doc_id)
            if token in document_list[index].term_dict:
                document_list[index].term_dict[token] += 1
            else:
                document_list[index].term_dict[token] = 1
        else:
            document_list.append(Document(doc_id))
    return


def normalize_term_frequency(term_dict):
    den = 0
    for term in term_dict:
        den += term_dict[term] ** 2
    den = sqrt(den)
    for term in term_dict:
        term_dict[term] /= den


def normalize_tf_idf(norm_file):  # Converts both tf and idf count to logarithm and normalizes entire tf-idf structure
    for document in document_list:
        for term in document.term_dict:
            document.term_dict[term] = 1 + log10(document.term_dict[term])
    for document in document_list:
        normalize_term_frequency(document.term_dict)
    for term in term_list:
        try:
            term.idf = log10(term.idf)
        except ValueError:
            print('idf is 0. exiting ...')
            exit(1)
    print('Normalizing process done')
    save_tf_idf(norm_file)
    return


def save_tf_idf(file_name):
    pickling_filename = file_name
    output = open(pickling_filename, 'wb')
    tf_idf = list()
    tf_idf.append(term_list)
    tf_idf.append(document_list)
    pickle.dump(tf_idf, output)
    print('Tf-Idf saved to file,', pickling_filename)
    return


def view_stats_tf_idf():
    for term in term_list:
        print('length', len(term.documents), 'term:', term, term.documents)
    for term in term_list:
        print('idf', term, term.idf)

for i in trange(i, len(files[:no_of_files_to_process])):
    filename = files[i]
    if i % 100 == 0:
        save_tf_idf(output_filename)
    with open(processed_songs_folder + filename, 'r') as f:
        content = f.read()
        words = word_tokenize(content)
        construct_tf_idf(words, filename[:-4])

save_tf_idf(output_filename)
normalize_tf_idf(normalized_filename)
# view_stats_tf_idf()
