#1

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import cx_Oracle

ip = 'ORBITOP'
port = 1521
SID = 'XE'
dsn_tns = cx_Oracle.makedsn(ip, port, SID)

connection = cx_Oracle.connect('orbitiumnos', 'legomania97', dsn_tns)

query = """SELECT*
           FROM spaces_dependence
           """
df_ora = pd.read_sql(query, con=connection)

connection.close()

G = nx.from_pandas_edgelist(df_ora, 'FATHER', 'CHILD')

#2

connection = cx_Oracle.connect('orbitiumnos', 'legomania97', dsn_tns)

query = """SELECT*
           FROM reglament_status
           """
df_status = pd.read_sql(query, con=connection)

connection.close()

def get_status(node):
    return df_status[df_status['STAGE']==node]['STATUS'].values[0]

color_map = []

for node in G:
    if get_status(node) == 'OK':
        color_map.append('green')
    elif get_status(node) == 'ERROR':
        color_map.append('red')
    else:
        color_map.append('white')
    print(node, get_status(node))

# 3

nx.draw(G, node_color = color_map, with_labels=True)
plt.show()