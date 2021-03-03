import pandas as pd
import numpy as np
import datetime as dt
import math

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#pegando a base de dados da web para se manter atualizado
original_df = pd.read_csv('https://github.com/owid/covid-19-data/blob/master/public/data/owid-covid-data.csv?raw=true')

#removendo os campos nos quais contem os dados do continente como um todo
df = original_df[original_df['continent'].notnull()]

df = df[['date', 'new_cases']]

#convertendo para o formato de data e agrupando por data
df['date'] = pd.to_datetime(df['date'], format = '%Y-%m-%dT')
df['date'] = df['date'].map(dt.datetime.toordinal)
df = (df.groupby(df['date']).sum()).reset_index()

#preenchendo os campos nulos de 'new_cases'
df['new_cases'] = df['new_cases'].fillna(0)


x = df['date']
y = df['new_cases']

#treinando o modelo
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
lreg = LinearRegression()
model = lreg.fit(np.array(x_train).reshape(-1, 1), np.array(y_train).reshape(-1, 1))

last_index = len(df) - 1
last_date = df.date[[last_index]]

#days = int(input("Digite o numero de dias a frente: "))


def prever(dias):
    d = 1
    while d <= dias:
        pred = model.predict(np.array([(last_date + d)]))
        number = math.trunc(pred[0, 0])
        print(d, '->', number)
        d += 1
    return


def verifica_input():
    days = int(input("Digite o numero de dias a frente: "))
    while days < 0:
        verifica_input()
    prever(days)

verifica_input()