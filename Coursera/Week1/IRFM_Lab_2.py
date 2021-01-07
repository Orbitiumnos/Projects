pip install pandas
pip install numpy
pip install csv
pip install cx_Oracle

#import matplotlib.pyplot as plt
#import math
#import scipy.stats as ss
#import seaborn as sns

#читаем данные с csv, кодировка обязательно нужна тк присутствует латинница
df = pd.read_csv("C:\Jypiter\Glob_terror.csv", encoding='latin1', error_bad_lines = False, warn_bad_lines = False, low_memory=False)
df