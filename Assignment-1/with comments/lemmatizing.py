from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import os
from tqdm import trange

songs_folder = './songs/'
output_folder = './processed_songs/' 
lemma_tzr = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))  # stop words so as to delete these from all files
files = os.listdir(songs_folder)  

pun_chars = ['!', '?', ':', '"', ',', '.', '[', ']', '{', '}', '\'', '`', '~', '@', '#', '$', '%', '^', '&', '*', '('
             ')', '+', '=', '|', '\\', '<', '>']
space_chars = ['-', '_']

for i in trange(len(files)): # looping through all songs
    filename = files[i]
    with open(songs_folder + filename, 'r') as f:
        sentence = f.read()
        tokens = word_tokenize(sentence)
        with open(output_folder + filename, 'w') as out:
            for token in tokens:
                token = lemma_tzr.lemmatize(token)  # Using WordNet Lemmatization
                token = stemmer.stem(token)  # Using PorterStemmer Method
                if token not in stop_words:  # ignores all stop words
                    token = "".join(c for c in token if c not in pun_chars) 
                    for c in token:
                        if c in space_chars:
                            i = token.index(c)
                            token = token[:i] + " " + token[i+1:]
                    out.write(token.lower() + ' ')
