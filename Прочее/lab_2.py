#!/opt/anaconda/envs/bd9/bin/python
import sys
import happybase

# Проверка на кол-во    
def check_length(lst):
    if len(lst) == 3:
        return True
    else: return False

# Проверка на соответствие ключу
def check_uid(uid):
    if int(uid) % 256 == 17:
        return True
    else: return False

# Проверка на тире
def check_tire(key):
    if str(key) != '-':
        return True
    else: return False

connection = happybase.Connection('spark-master.newprolab.com')
table = connection.table('nikolay.fedorov')    
    
for line in sys.stdin:    
    # создание списка и очистка от пустых элементов
    lst = line.split('\t')
    lst = ' '.join(lst).split()
    # проверки
    if check_length(lst) is True:
        if check_tire(lst[1]) is True:
            if check_tire(lst[0]) is True:
                if check_uid(lst[0]) is True:
                    #запись в базу
                    key = lst[0]
                    timest = lst[1]
                    url = lst[2]
                    table.put(key, {'data:url': url}, timestamp=int(float(timest)*1000))