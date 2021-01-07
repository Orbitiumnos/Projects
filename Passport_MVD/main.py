import requests #импортируем модуль
import sys, time, bz2, os, argparse
import multiprocessing as mp
from datetime import datetime
from functools import partial
from math import ceil
from urllib.request import urlopen

'''
f=open(r'C:\file_bdseo.zip',"wb") #открываем файл для записи, в режиме wb
ufr = requests.get("http://site.ru/file.zip") #делаем запрос
f.write(ufr.content) #записываем содержимое в файл; как видите - content запроса
f.close()

http://guvm.mvd.ru/upload/expired-passports/list_of_expired_passports.csv.bz2

import requests #импортируем модуль
#f=open(r'D:\file_bdseo.zip',"wb") #открываем файл для записи, в режиме wb
ufr = requests.get('http://guvm.mvd.ru/upload/expired-passports/list_of_expired_passports.csv.bz2') #делаем запрос
f.write(ufr.content) #записываем содержимое в файл; как видите - content запроса
f.close()
'''


filename = r'C:\Users\Николай\Documents\SCRIPTS\Files for Notebooks and Scripts\MVD.txt'
#req = urllib2.urlopen('http://example.com/file.bz2')
req = requests.get('http://guvm.mvd.ru/upload/expired-passports/list_of_expired_passports.csv.bz2', stream=True)
CHUNK = 16 * 1024

decompressor = bz2.BZ2Decompressor()
with open(filename, 'wb') as fp:
    while True:
        chunk = req.read(CHUNK)
        if not chunk:
            break
        fp.write(decompressor.decompress(chunk))
req.close()

target_path = r'C:\Users\Николай\Documents\SCRIPTS\Files for Notebooks and Scripts'
url = 'http://guvm.mvd.ru/upload/expired-passports/list_of_expired_passports.csv.bz2'
response = requests.get(url, stream=True)
handle = open(target_path, "wb")
for chunk in response.iter_content(chunk_size=512):
    if chunk:  # filter out keep-alive new chunks
        handle.write(chunk)

#-----------------------------------------------------------------------------------------------------------------------

def download_file(url):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk: # filter out keep-alive new chunks
                    print('writing')
                    f.write(chunk)
                    # f.flush()
    print('end')
    return local_filename

download_file(url)

import os
statinfo = os.stat('list_of_expired_passports.csv.bz2')
print(statinfo.st_size)

with bz2.open("myfile.bz2", "rb") as f:
...     # Decompress data from file
...     content = f.read()


'''
for filename in files:
    filepath = os.path.join(dirpath, filename)
    newfilepath = os.path.join(dirpath,filename + '.decompressed')
    with open(newfilepath, 'wb') as new_file, open(filepath, 'rb') as file:
        decompressor = BZ2Decompressor()
        for data in iter(lambda : file.read(100 * 1024), b''):
            new_file.write(decompressor.decompress(data))
'''

import os
dirpath = r'C:\Users\Николай\PycharmProject\Passport_MVD'
filename = 'list_of_expired_passports.csv.bz2'

filepath = os.path.join(dirpath, filename)
newfilepath = os.path.join(dirpath, filename + '.decompressed')
with open(newfilepath, 'wb') as new_file, bz2.BZ2File(filepath, 'rb') as file:
    for data in iter(lambda : file.read(100 * 1024), b''):
        new_file.write(data)

print('hello world')