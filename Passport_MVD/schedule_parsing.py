#1

import re

path = r'C:\Data for Scripts\schedule.html'

list = []

def cleanhtml(raw_html):
  cleanr = re.compile('(<.*?>|[\nxa])')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

#f = open(r'C:\Data for Scripts\schedule.html', 'r', encoding )
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        result = cleanhtml(line)
        list.append(result)
print(list)

text = "\n".join(list)
print(text)

#2
import bleach
list = []

#file = open(r'C:\Data for Scripts\schedule.html', 'r', encoding='utf-8')
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        list.append(list)

str = "\n".join(list)

res = bleach.clean(str, tags=[], attributes={}, styles=[], strip=True)
print(res)

#3

beautifulsoup
