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
from arch import arch_model


def Page(side_head,head,Select_box_option):
    start = '2007-03-09'
    end = dt.datetime.today().strftime('%Y-%m-%d')
    st.set_page_config(layout="wide")
    st.header(head)
    st.sidebar.header(side_head)
    Index_Select_Box = st.sidebar.selectbox(label = "",
    options = Select_box_option.index,key="World_Indices") 

    Movig_Avg_selectbox = st.sidebar.selectbox(label = "Moving Average",
    options = "7 14 21 30 50 100 200 300".split(),key="MA")
    return Movig_Avg_selectbox,Index_Select_Box,start,end

def Layout(tit,bgcol):
    layout = go.Layout(
        title=tit,    
        plot_bgcolor=bgcol,
        hovermode="x",
        hoverdistance=100, # Distance to show hover label of data point
        spikedistance=1000,
        legend=dict(
            # Adjust click behavior
            itemclick="toggleothers",
            itemdoubleclick="toggle"
        ),
        xaxis=dict(
            title="Date",
            linecolor="#BCCCDC",
            showgrid=False,
            spikethickness=2,
            spikedash="dot",
            spikecolor="#999999",
            spikemode="across"
        ),
        yaxis=dict(
            title="Price",
            linecolor="#BCCCDC",
            showgrid=False,
            spikethickness=2,
            spikedash="dot",
            spikecolor="#888888",
            spikemode="across"
        ),height = 600,
        width = 600
    )
    return layout


def MAChart(Selected_Moving_Avg,dataframe,chart_title,chart_bg):

    MA = dataframe['Adj Close'].rolling(int(Selected_Moving_Avg)).mean().dropna()

    Modified_dataframe_chart = go.Scatter(
                x=MA.index,
                y=MA.values,
                name = str(Selected_Moving_Avg)+" MA",
                marker_color = "#FFFFFF"
            )

    Original_dataframe_chart = go.Scatter(
                x=dataframe.index[int(Selected_Moving_Avg):],
                y=dataframe['Adj Close'].values[int(Selected_Moving_Avg):], #Adj Close
                name = "Adj Close",
                marker_color='#FF6700'
                
            )

    Figures = [Modified_dataframe_chart ,Original_dataframe_chart]
    Moving_Average_Plot = go.Figure(data=Figures, layout=Layout(chart_title,chart_bg))
    st.plotly_chart(Moving_Average_Plot,use_container_width=True,sharing="streamlit")

def Cluster(df,Garch_pq,Stock):
    Lstm_Cluster = df.loc[Stock][1]
    Garch_Cluster = df.loc[Stock][2]
    pq =  Garch_pq[Garch_Cluster]
    return Lstm_Cluster, Garch_Cluster, pq

def GARCH_Chart(name,Original,Predicted,chart_title,chart_bg,p,q):
    st.header("Volatality & Daily Returns")

    st.subheader("Prediction")
    Daily_Ret = Original['Adj Close'].pct_change().dropna() #Adj Close
    Modified_dataframe_chart = go.Scatter(
                x=Daily_Ret.index[-len(Predicted):],
                y=Daily_Ret.values[-len(Predicted):],
                name = "Daily Returns",
            )

    Original_dataframe_chart = go.Scatter(
                x=Daily_Ret.index[-len(Predicted):],
                y=Predicted/100,
                name = "Predicted Volatality",
                marker_color='#FFFFFF'
                
            )
            

    Figures = [Modified_dataframe_chart ,Original_dataframe_chart]
    Vola_Plot = go.Figure(data=Figures, layout=Layout(chart_title,chart_bg))
    st.plotly_chart(Vola_Plot,use_container_width=True,sharing="streamlit")
    datelist = pd.bdate_range(dt.date.today(), periods=7).tolist()
    Dt = [ str(x).split()[0] +" | " + str(x.strftime('%A')) for x in datelist]
    Daily_Ret = Daily_Ret*100


    st.subheader("Forecast")
    Dt = [ str(x).split()[0] +" | " + str(x.strftime('%A')) for x in datelist]
    D = [ str(x).split()[0]  for x in datelist]
    GA = st.checkbox('Table',False,key = 'GT')
    am = arch_model(Daily_Ret, vol="Garch", p=p, q=q, dist="Normal")
    res = am.fit(disp='off')
    forecasts = res.forecast(reindex=False,horizon=7)
    Pred = pd.Series(np.sqrt(forecasts.variance.values[-1,:]), index=Dt, name = "Magnitude of Volatility")
    if GA:
        st.table(Pred/100)
    else:
        chart = go.Scatter(
                x=D,
                y=Pred/100,
                name = "Forecasted Volatality",
                marker_color='#FFFFFF')
        Vol = [chart]
        vol = go.Figure(data=Vol, layout=Layout(chart_title,chart_bg))
        st.plotly_chart(vol,use_container_width=True,sharing="streamlit")
