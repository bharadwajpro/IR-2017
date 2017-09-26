from bs4 import BeautifulSoup
from os import listdir
from os.path import isfile, join

folder = '../sitemap/'
files = [f for f in listdir(folder) if isfile(join(folder, f))]
link_no = 1

with open('links.txt', 'w') as fp:
    for f in files:
        with open(folder + f, 'r') as xml:
            soup = BeautifulSoup(xml, 'xml')
            links = soup.find_all('loc')
            for link in links:
                link = link.get_text()
                if 'song' in link:
                    fp.writelines(link + '\n')
                    print('\r%d links wrote' % link_no, end='')
                    link_no += 1
