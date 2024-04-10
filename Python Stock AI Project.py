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

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds= pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis = 1)
    return combined


#training the model with x years of data
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

predictions = backtest(sp500, model, predictors)

values1 = predictions["Predictions"].value_counts()

print(values1)

precision_score(predictions["Target"], predictions["Predictions"])

predictions["Target"].value_counts() / predictions.shape[0]

#adding addtional predictors to the model

horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
    
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors += [ratio_column, trend_column]
    
sp500 =sp500.dropna()

sp500

#improving the model 

model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0 
    preds= pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis = 1)
    return combined


predictions = backtest(sp500, model, new_predictors)

predictions["Predictions"].value_counts()


# 0.0 output will tell us the days price went down
# 1.0 output will tell us the days price went up 

precision_score(predictions["Target"], predictions["Predictions"])


#output the higher the better and more accuarte of data


#things to do with model 
#