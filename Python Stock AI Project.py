#Version 1
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import yfinance as yf 
#calls api to fetcxh data

#calling sp500 stock data
sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")

#cleaning and visualizing stock market data

sp500.plot.line(y="Close", use_index=True)
plt.show()



