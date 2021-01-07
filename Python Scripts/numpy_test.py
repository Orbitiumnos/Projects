

# Import pandas library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
# initialize list of lists
data = [['tom', 10], ['nick', 15], ['juli', 14]]
# Create the pandas DataFrame
df = pd.DataFrame(data, columns = ['Name', 'Age'])
# print dataframe.
df
'''

data = pd.read_excel(r'C:\Users\Nikolay\Google Диск\9 семестр\НИР\dataset.xlsx', 'LIKV2', parse_dates=True, dayfirst=True, index_col= 0, names=['Date', 'Nal','Ost','Rez'])

fig = plt.figure(figsize=(12, 18))

plt.subplot(3, 1, 1)
fig = plt.plot(data['Nal'], label="Nal", marker='', c='r')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
fig = plt.plot(data['Ost'], label="Ost", marker='', c='g')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
fig = plt.plot(data['Rez'], label="Rez", marker='', c='b')
plt.legend()
plt.grid(True)

plt.show()

pip install pandas