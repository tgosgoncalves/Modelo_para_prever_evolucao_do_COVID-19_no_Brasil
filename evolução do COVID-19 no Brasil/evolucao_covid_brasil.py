import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import re


df = pd.read_csv("covid_19_data.csv", parse_dates=['ObservationDate', 'Last Update'])
df.head()

df.dtypes

def corrigir_colunas(col_name):
    return re.sub(r"[/| ]", "", col_name).lower()

df.columns = [corrigir_colunas(col) for col in df.columns]
df.head()

df.loc[df.countryregion == 'Brazil']

brasil = df.loc[
    (df.countryregion == 'Brazil') &
    (df.confirmed >0)
]

brasil.head()

px.line(brasil, 'observationdate', 'confirmed', title = 'Casos Confirmados no Brasil')

brasil['novoscasos'] = list(map(
    lambda x: 0 if (x==0) else brasil['confirmed'].iloc[x] - brasil['confirmed'].iloc[x-1],
    np.arange(brasil.shape[0])
))

px.line(brasil, x='observationdate', y='novoscasos', title='Novos casos por dia')

fig = go.Figure()

fig.add_trace(
    go.Scatter(x=brasil.observationdate, y=brasil.deaths, name='Mortes',
               mode='lines+markers', line={'color':'red'})
)


fig.update_layout(title='Mortes por COVID-19 no Brasil')

fig.show()

def taxa_crescimento(dados, variavel, data_inicio=None, data_fim=None):
    if data_inicio == None:
        data_inicio = dados.observationdate.loc[dados[variavel] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)
    
    if data_fim == None:
        data_fim = dados.observationdate.iloc[-1]
    else:
        data_fim = pd.to_datetime(data_fim)
        
        
    passado = dados.loc[dados.observationdate == data_inicio, variavel].values[0]
    presente = dados.loc[dados.observationdate == data_fim, variavel].values[0]
    
    
    n = (data_fim - data_inicio).days
    

    taxa = (presente/passado)**(1/n) - 1
    
    return taxa*100

taxa_crescimento(brasil, 'confirmed')

def taxa_crescimento_diaria(dados, variavel, data_inicio=None):
    if data_inicio == None:
        data_inicio = dados.observationdate.loc[dados[variavel] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)
        
    data_fim = dados.observationdate.max()
    
    
    n = (data_fim - data_inicio).days
    
    
    taxas = list(map(
        lambda x: (dados[variavel].iloc[x] - dados[variavel].iloc[x-1]) / dados[variavel].iloc[x-1],
        range(1, n+1)
    ))
    return np.array(taxas) * 100

    tx_dia = taxa_crescimento_diaria(brasil, 'confirmed')

    tx_dia

    primeiro_dia = brasil.observationdate.loc[brasil.confirmed >0].min(0)

px.line(x=pd.date_range(primeiro_dia, brasil.observationdate.max())[1:],
        y=tx_dia, title='Taxa de crescimento de casos confirmados no Brasil')


from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

confirmados = brasil.confirmed
confirmados.index = brasil.observationdate
confirmados

res = seasonal_decompose(confirmados)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10,8))

ax1.plot(res.observed)
ax2.plot(res.trend)
ax3.plot(res.seasonal)
ax4.plot(confirmados.index, res.resid)
ax4.axhline(0, linestyle='dashed', c='black')
plt.show()    

mortes = brasil.deaths
mortes.index = brasil.observationdate
mortes

res = seasonal_decompose(mortes)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10,8))

ax1.plot(res.observed)
ax2.plot(res.trend)
ax3.plot(res.seasonal)
ax4.plot(mortes.index, res.resid)
ax4.axhline(0, linestyle='dashed', c='black')
plt.show()      

from pmdarima.arima import auto_arima
modelo = auto_arima(confirmados)

fig = go.Figure(go.Scatter(
    x=confirmados.index, y=confirmados, name='Observados'
))

fig.add_trace(go.Scatter(
    x=confirmados.index, y=modelo.predict_in_sample(), name='Preditos'
))

fig.add_trace(go.Scatter(
    x=pd.date_range('2020-05-20', '2020-06-20'), y=modelo.predict(31), name='Forecast'
))

fig.update_layout(title='Previsão de casos confirmados no Brasil para os próximos 30 dias')
fig.show()

modelo = auto_arima(mortes)

fig = go.Figure(go.Scatter(
    x=mortes.index, y=mortes, name='Observados'
))

fig.add_trace(go.Scatter(
    x=mortes.index, y=modelo.predict_in_sample(), name='Preditos'
))

fig.add_trace(go.Scatter(
    x=pd.date_range('2020-05-20', '2020-06-30'), y=modelo.predict(31), name='Forecast'
))

fig.update_layout(title='Previsão de mortes no Brasil para os próximos 30 dias')
fig.show()

