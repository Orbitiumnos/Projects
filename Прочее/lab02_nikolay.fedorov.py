#mapper
#!/opt/anaconda/envs/bd9/bin/python

import sys
def emit(key, value, url):
    sys.stdout.write('{}\t{}\t{}\n'.format(key, value, url))

def Map(line):
    objects = line.split('\t')
    if len(objects) == 3:
        UID, timestamp, URL = objects
        # Николай Ф: UID и uid - это две разные переменные
        # Николай Ф: Также отсутствует проверка на тире, что вызывает ошибку неверного формата
        if URL.startswith('http') and uid % 256 == 154:
            UID = int(UID)
            timestamp = int(float(timestamp) * 1000)
            emit(UID, timestamp, URL)

# Николай Ф: нельзя вызывать метод вместе с пар-ом который не инициирован
def main(line):
    for line in sys.stdin:
        Map(line.strip())

if __name__ == '__main__':
    main()

#reducer
import happybase
connection = happybase.Connection('spark-master.newprolab.com')
connection.create_table(
    'mytable',
    {'cf1': dict(max_versions=10),
    }
)
table = connection.table('mytable')
for line in sys.stdin:
    row = line.strip().split('\t')
    # Николай Ф: на выходе mapperа в строке сначала стоит t, потом url 
    uid, url, t = row
    table.put(uid, {b'cf1:col1': url}, timestamp=int(t))