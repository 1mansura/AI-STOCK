#Version 1
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import yfinance as yf 
import pandas


#calls api to fetch data
#calling sp500 stock data
sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")

#cleaning and visualizing stock market data
plot = sp500.plot.line(y="Close", use_index=True)

# Show the plot
plt.show()

#delete data that is not needed from stock
del sp500["Dividends"]
del sp500["Stock Splits"]

#Setting up target for the machine learning
sp500["Tomorrow"] = sp500["Close"].shift(-1)

# Print the data
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

sp500 = sp500.loc["1990-01-01":].copy()

#training an initial machine learning model

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])

RandomForestClassifier(min_samples_split=100, random_state=1)

from sklearn.metrics import precision_score

import pandas as pd
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

precision_score(test["Target"], preds)

combined = pd.concat([test["Target"], preds], axis=1)

combined.plot()
plt.show()
#accuary metric



#building a back test system
