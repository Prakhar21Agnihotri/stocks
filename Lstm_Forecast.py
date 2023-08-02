from ast import Mod
from pyparsing import original_text_for
import streamlit as st
import plotly.graph_objs as go
import pandas_datareader.data as web
import datetime as dt
import plotly.express as px
import tensorflow as tf
import numpy as np
import pandas as pd
from Func import Layout
import datetime as dt

Lstm_Models = [tf.keras.models.load_model('Cluster_0_LSTM.h5'),
tf.keras.models.load_model('Cluster_1_LSTM.h5'),
tf.keras.models.load_model('cluster02.h5'),
tf.keras.models.load_model('Cluster_3_LSTM.h5'),
tf.keras.models.load_model('Cluster_4_LSTM.h5'),
tf.keras.models.load_model('cluster5.h5'),
tf.keras.models.load_model('cluster6.h5'),
tf.keras.models.load_model('cluster7.h5'),
tf.keras.models.load_model('cluster8n.h5'),
tf.keras.models.load_model('cluster9.h5')]

def LSTM(name,Prediction,Original,bgcol,model):
    st.header("Prediction & Forecast")
    if model<5:
        Modified_dataframe_chart = go.Scatter(
                        x=Original.index[30:],
                        y=Prediction.reshape(-1),
                        name = "Predicted Adjusted Closing Price", #Predicted Adjusted Closing Price
                        marker_color = '#FFFFFF'
                    )

        Original_dataframe_chart = go.Scatter(
                        x=Original.index[30:],
                        y=Original['Close'].values[30:],
                        name = "Daily Closing Price", #Daily Adjusted Closing Price
                        marker_color='#996699'
                        
                    )
        

    
    else:
        Modified_dataframe_chart = go.Scatter(
                        x=Original.index[100:],
                        y=Prediction.reshape(-1),
                        name = "Predicted Closing Price", #Predicted Adjusted Closing Price
                        marker_color = '#FFFFFF'
                    )

        Original_dataframe_chart = go.Scatter(
                        x=Original.index[100:],
                        y=Original['Adj Close'].values[100:],
                        name = "Daily Closing Price", #Daily Adjusted Closing Price
                        marker_color='#996699'
                        
                    )
        

    Figures = [Modified_dataframe_chart ,Original_dataframe_chart]
    Lstm_plot = go.Figure(data=Figures,layout=Layout(name,bgcol))
    
    st.plotly_chart(Lstm_plot,use_container_width=True,sharing="streamlit")
    
    if model<5:
        T=Original['Close'].values[-30:] #'Adj Close'
    else:
        T=Original['Close'].values[-100:] #'Adj Close'

    for c in range(T.shape[0],T.shape[0]+30):
        l = Forecast(T,c,model)
        list(T).append(l)
        np.array(T)
    
    Forecasted = T[-30:]
    datelist = pd.bdate_range(dt.date.today(), periods=30).tolist()
    Dt = [ str(x).split()[0] +" | " + str(x.strftime('%A')) for x in datelist]
    Table = pd.Series(Forecasted,Dt,name = "Closing Price") # "Adjusted Closing Price"
    st.subheader("{} Days Forecast".format(len(Forecasted)))

    Forecast_Chart = go.Scatter(
                        x=datelist,
                        y=Forecasted,
                        name = name,
                        marker_color='#996699'
                        
                    )
    Fig = [Forecast_Chart]
    Forecast_plot = go.Figure(data=Fig,layout=Layout(name,bgcol))
    TA = st.checkbox('Table',False,key='LT')
    if TA:
        st.table(Table)
    else:
        st.plotly_chart(Forecast_plot,use_container_width=True,sharing="streamlit")

def Forecast(data,j,model):
    d = data[-(data.shape[0]):].reshape(-1,1)
    model = Lstm_Models[model]
    from sklearn.preprocessing import MinMaxScaler
    Min = MinMaxScaler(feature_range=(0,1))
    fore = Min.fit_transform(d)
    fore = fore.reshape(-1,d.shape[0],1)
    p = model.predict(fore)
    q = Min.inverse_transform(p).flatten()
    return q

