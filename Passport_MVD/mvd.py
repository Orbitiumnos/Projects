# Проверяет состоит ли строка только из цифр
def isInteger(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


# Размер блока чтения (в строках). Больше значение - Больше расход RAM
blocksize = 20 * 10 ** 6  # 1200MB


# Переводит RAM в размер блока в строках
def toBlock(ram):
    global blocksize
    mblock = 600  # 5m ~ 600MB
    isGB = ram.find('GB')
    isMB = ram.find('MB')
    if isGB == isMB:
        print('Error in ram_use variable:', ram_use, 'Using default RAM')
        print('Example: \'1GB 200MB\'')
        logging('Using default RAM')
        ram = '1GB'
        isGB = 1
    print('RAM USING:', ram)
    logging('RAM USING: ' + ram)
    sizeGB = 0
    sizeMB = 0
    if isGB != -1:
        partGB, ram = ram.split('GB')
        sizeGB = int(partGB)
    if isMB != -1:
        partMB, ram = ram.split('MB')
        sizeMB = int(partMB)
    size = sizeGB * 2 ** 10 + sizeMB
    blocksize = int(size / mblock * (5 * 10 ** 6))
    print('Blocksize computed: ' + str(blocksize // 10 ** 6) + 'm passports!')
    logging('Blocksize computed: ' + str(blocksize // 10 ** 6) + 'm passports!')


# Конвертирует файл с номерами паспортов в формат для загрузки в Кронос
def formatKronos(file, name):
    print('Converting File to Kronos format')
    logging('Converting File to Kronos format')
    start_package = '++ ДД'  # начало пакета
    end_package = '++ ЯЯ'  # конец пакета
    start_message = '++ НН'  # начало сообщения
    end_message = '++ КК'  # конец сообщения
    div = '‡'  # разделитель
    mnemo_code = '++ МП'  # мнемокод базы
    # Начало строки
    start_ = start_package + div + start_message + div + mnemo_code + div
    # Конец строки
    _end = div + end_message + div + end_package + div
    with open(file, 'r') as fd, open(name + postfix, 'w') as kron:
        print(file + ' converting to ' + name + postfix)
        logging(file + ' converting to ' + name + postfix)
        file_len = sum(1 for n in fd)
        fd.seek(0)
        for k, line in enumerate(fd):
            kron.write(start_ + '01 ' + line[:4] + div + '02 ' + line[4:10] + _end)
            if k < file_len - 1:
                kron.write('\n')
            if k % 1000 == 0:
                print(str(k * 100 // file_len) + '%', end='\r')
    print('Converted!')
    logging('Converted!')


# Запись в лог
def logging(text, noTime=0):
    with open('./log/log' + postfix, 'a') as log:
        if noTime:
            print(text, file=log)
        else:
            print(datetime.today().strftime('[%Y-%m-%d %H:%M:%S] ') + text, file=log)


# Скачивание файла по ссылке
def downloadFile(url):
    filename = url.split('/')[-1]
    print('Downloading:', filename)
    logging('Downloading: ' + filename)
    # Если файл уже существует - пропуск
    if os.path.exists(filename):
        print(filename, 'exists! Skipped!')
        logging(filename + ' exists! Skipped!')
        return filename

    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        size = 0
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                if size % 10240 == 0:
                    print('Downloaded:', str(size // 1024) + 'MB', end='\r')
                f.write(chunk)
                f.flush()
                size += 1
    print('Downloaded:', filename, str(size // 1024) + 'MB')
    logging('Downloaded: ' + filename + ' ' + str(size // 1024) + 'MB')
    return filename


# Разархивирование bz2
def decompressFile(filename='list_of_expired_passports.csv.bz2'):
    print('Extracting:', filename)
    logging('Extracting: ' + filename)
    # Если файл уже существует - пропуск
    if os.path.exists(filename[:-len(fformat)]):
        print(filename[:-len(fformat)], 'exists! Skipped!')
        logging(filename[:-len(fformat)] + ' exists! Skipped!')
        return filename[:-len(fformat)]

    with open(filename[:-len(fformat)], 'wb') as csvfile, open(filename, 'rb') as zipfile:
        z = bz2.BZ2Decompressor()
        for block in iter(lambda: zipfile.read(512 * 1024), b''):
            csvfile.write(z.decompress(block))
    print('Extracted', filename[:-len(fformat)])
    logging('Extracted ' + filename[:-len(fformat)])
    return filename[:-len(fformat)]


# Удаление всех данных кроме вида: 1234,123456 (считаются ошибочными)
def parseCSV(filename='list_of_expired_passports.csv'):
    print('Parsing:', filename)
    logging('Parsing ' + filename)
    pfilename = filename[:-len(fformat)] + postfix
    num = 0
    err = 0
    # Если файл уже существует - пропуск
    if os.path.exists(pfilename):
        with open(pfilename, 'r') as pfile:
            num = sum(1 for i in pfile)
        print(pfilename, 'exists!', num, 'passports! Skipped!')
        logging(pfilename + ' exists! ' + str(num) + ' passports! Skipped!')
        return num, pfilename

    with open(filename, 'r', encoding='utf8') as csvIN, \
            open(pfilename, 'w') as txtOUT, \
            open('brokenData.txt', 'w') as txtBroke:
        next(csvIN)
        for line in csvIN:
            a, b = line.replace('\n', '').split(',')
            if len(a) == 4 and len(b) == 6 and (a + b).isdigit():
                txtOUT.write(a + b + '\n')
                num += 1
                if num % 10 ** 5 == 0:
                    print('Passports:', num, end='\r')
            else:
                err += 1
                txtBroke.write(a + ',' + b + '\n')
        print('Parsed', num, 'passports!')
        print('File:', pfilename)
        print('Broken Data: brokenData.txt (' + str(err) + ')')
        logging('Parsed ' + str(num) + ' passports!\nFile: ' +
                pfilename + '\nBroken Data: brokenData.txt (' + str(err) + ')')
        return num, pfilename


# Поиск в директории ./backup самого последнего файла по postfix дате
def getBackFile(filename='list_of_expired_passports.csv'):
    print('Getting backup file to compare')
    logging('Getting backup file to compare')
    n = len(postfix) - 1
    flen = len(fformat)
    f = []
    for root, dirs, files in os.walk('./backup'):
        f.extend(files)
        break
    if len(f) == 0:
        print('No backup files! Set \'pure_start = 1\' Abort.')
        logging('No backup files! Set \'pure_start = 1\' Abort.')
        exit()
    last = 0  # последний бэкап
    first = 0  # первый бэкап
    for file in f:
        end_f = file[-n:-flen]
        if not isInteger(end_f):
            print('Postfix error: not a number! Abort.', end_f)
            logging('Postfix error: not a number! Abort. ' + end_f)
            exit()
        if last < int(end_f):
            last = int(end_f)
            first = last if first == 0 else first
        if first > int(end_f):
            first = int(end_f)
    print('Got first backup:', first)
    print('Got last backup:', last)
    logging('Got first backup: ' + str(first) +
            ' Got last backup: ' + str(last))
    return (filename[:-flen] + '_' + str(first) + fformat), (filename[:-flen] + '_' + str(last) + fformat)


# Переводит строку в необходимый формат для записи в стек
def setFormat(line):
    if line[0] == '0':
        return line.replace('\n', '')
    return int(line)


_____________________________________________________________________________________________________________________________________


# Тело основной программы
def main():
    print('Starts passports parser!')
    t0 = time.time()

    # Инициализация
    init()

    # Скачиваем реестр недействительных паспортов
    compressfile = downloadFile(fms_url)
    # Распаковываем архив в текущую директорию
    first_backup = file = decompressFile(compressfile)
    # Подчищаем файл от битых данных
    num_passports, parsed_file = parseCSV(file)
    # Если запуск первый, то сохранить только бэкап
    if not pure_start:
        # Получение имени предыдущей версии реестра для вычисления дельты
        first_backup, backup_file = getBackFile(file)
        # Сравнение старой и новой версии баз, выделение дельты
        calcDelta(backup_file, parsed_file, num_passports)
        # Конвертирование в формат Кроноса
        if kronos:
            # Если файлы существуют
            if delta_type == 'plus' or delta_type == 'all':
                formatKronos('deltaPlus' + postfix, 'kronos_add')
            if delta_type == 'minus' or delta_type == 'all':
                formatKronos('deltaMinus' + postfix, 'kronos_del')

    t1 = time.time()
    print('Parser ended!')
    print('Time: ', '{:g}'.format((t1 - t0) // 60), 'm ', '{:.0f}'.format((t1 - t0) % 60), 's', sep='')
    logging('---------------\nCompleted!', 1)
    logging('Time: ' + str('{:g}'.format((t1 - t0) // 60)) + 'm ' + str('{:.0f}'.format((t1 - t0) % 60)) + 's', 1)

    # Постобработка - завершение
    postprocessing(parsed_file, first_backup, file, compressfile)


if __name__ == '__main__':
    mp.freeze_support()  # фикс ошибки с pyinstaller и multiprocessing и argparser
    main()