from bs4 import BeautifulSoup
import requests
from requests.exceptions import ConnectionError, ChunkedEncodingError
from langdetect import detect
import re
from tqdm import trange
import os

songs_folder = '../songs/'
files_skipped = 0
try:
    with open('songs_done.txt', 'r') as done_file:
        count = done_file.readline()
        count = int(count)
except FileNotFoundError:
    count = -1
with open('links.txt', 'r') as f:
    links = f.readlines()
    for i in trange(len(links)):
        if i <= count:
            continue
        link = links[i]
        try:
            html = requests.get(link).text
        except ConnectionError or ChunkedEncodingError:
            continue
        soup = BeautifulSoup(html, 'html.parser')
        song = soup.find_all('p', {'class': 'songtext'})[0]
        song_filtered = song.get_text()
        try:
            lang = detect(song_filtered[:50])
        except Exception:
            continue
        if lang != 'en':
            with open('songs_done.txt', 'w') as done_file:
                done_file.write(str(i))
            continue
        song_title = re.findall(r'/s.*?.html', link)[0][1:-5]
        try:
            with open(songs_folder + song_title + '.txt', 'w') as sf:
                sf.write(song_filtered)
        except UnicodeEncodeError as err:
            print(song_title + lang)
            print('%d files skipped' % (files_skipped + 1))
            files_skipped += 1
            try:
                os.remove(songs_folder + song_title + '.txt')
            except FileNotFoundError:
                print('Skipped file is not deleted because it is not there')
        with open('songs_done.txt', 'w') as done_file:
            done_file.write(str(i))
