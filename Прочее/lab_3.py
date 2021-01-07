#!/opt/anaconda/envs/bd9/bin/python

from urllib.parse import urlparse, unquote
import re
import sys


# парсер url
def url2domain(url):
    try:
        a = urlparse(unquote(url.strip()))
        if (a.scheme in ['http','https']):
            b = re.search("(?:www\.)?(.*)",a.netloc).group(1)
            if b is not None:
                return str(b).strip()
            else:
                return ''
        else:
            return ''
    except:
        return


# вывод
def emit(key, cat, mcat):
    sys.stdout.write('{}\t{}\t{}\n'.format(key, cat, mcat))


# проверка на тире
def check_tire(id):
    if str(id) != '-':
        return True
    else: return False


# проверка на http
def check_http(url):
    if url.startswith('http'):
        return True
    else: return False

# присвоение категории
def find_category(url):

    d = {
        'cars.ru': 11, 
        'avto-russia.ru': 12,
        'bmwclub.ru': 13, 
        'fastpic.ru': 21,
        'fotoshkola.net': 22, 
        'bigpicture.ru': 23,
        'nirvana.fm': 31, 
        'rusradio.ru': 32,
        'pop-music.ru': 33, 
        'snowmobile.ru': 41,
        'nastroisam.ru': 42, 
        'mobyware.ru': 43
    }
    url = url2domain(url)  
    try:
        if d.get(url,'None') == 'None': 
            return 'None','None','None'
        else:
            return str(d[url])[0], d[url], url
    except:
        return

# маппер, на вход - строка, выход - uid и категории
def Map(line):
    objects = line.strip().split('\t')
    cat = 0 
    mcat = 0
    if len(objects) == 3:
        uid, timestamp, url = objects
        if check_tire(uid) == True:
            if check_tire(url) == True:
                if check_http(url) == True:
                    mcat, cat, url = find_category(url)
                    if mcat != 'None':
                        #print(uid, mcat, cat)
                        emit(uid, mcat, cat)

# основная функция
def main():
    for line in sys.stdin:
        Map(line)

if __name__ == '__main__':
    main()