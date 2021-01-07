#mapper.py
import sys

def do_map(record):
    lst = record.split(', ')
    return lst[2], lst[4]

for line in sys.stdin:
    key, value = do_map(line)
    print(key + '\t' + str(value))
    list.append(key + '\t' + str(value))