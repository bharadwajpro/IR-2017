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
