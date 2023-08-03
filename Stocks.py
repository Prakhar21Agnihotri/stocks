import datetime as dt
from pandas_datareader import data as web
import yfinance as yf
yf.pdr_override()
b = yf.download(tickers = ['^GSPC'], start = '2007-03-09', end = dt.datetime.today().strftime('%Y-%m-%d'))
from Func import *
from Lstm_Forecast import*

Stocks = pd.read_pickle('Stocks_Name')
Garch_pq = pd.read_pickle('Garch_pq')
Predicted_price = pd.read_pickle("Predicted_data.pkl")
Predicted_Vola = pd.read_pickle("Garch_Predicted.pkl")



Selected = dict()
Selected_LSTM = dict()
Selected_Movig_Avg, Selected_Stock, start, end = Page("Stocks","Stocks",Stocks)
if Selected_Stock not in Selected.keys():
    Selected[Selected_Stock] = web.DataReader(Stocks.loc[Selected_Stock][0], 
    data_source = 'yahoo',start= start , end = end)



Lstm ,Garch, pq = Cluster(Stocks,Garch_pq,Selected_Stock)

MAChart(Selected_Movig_Avg, Selected[Selected_Stock], Selected_Stock,"#333333")

LSTM(Selected_Stock, Predicted_price[Selected_Stock], Selected[Selected_Stock],"#333333",Lstm)

GARCH_Chart(Selected_Stock,Selected[Selected_Stock],Predicted_Vola[Selected_Stock],Selected_Stock,"#333333",pq[0],pq[1])
