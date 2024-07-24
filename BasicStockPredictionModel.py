import math, datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

# Read CSV and set index
df = pd.read_csv(r'C:\Users\91904\OneDrive\Desktop\GOOGL.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)

df['HL_PCT'] = (df['High']-df['Close'])/df['Close']*100
df['PCT_change'] = (df['Close']-df['Open'])/df['Open']*100
df = df[['Close','HL_PCT','PCT_change','Volume']]

forecast_col='Close'
df.fillna(-99999,inplace=True)
forecast_out = int(math.ceil(0.001*len(df))) 
print(forecast_out)

df['Label'] = df[forecast_col].shift(-forecast_out)
print(df)

X = np.array(df.drop(columns=['Label']))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]
df.dropna(inplace=True)
y = np.array(df['Label'])
print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
with open('linearRegression.pkl','wb') as f:
   pickle.dump(clf, f)

pickle_in = open('linearRegression.pkl','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
predicted_value = clf.predict(X_lately)
print(predicted_value, accuracy, forecast_out)

df['pred_value'] = np.nan

last_date = df.index[-1]
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in predicted_value:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Close'].plot()
df['pred_value'].plot()
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend(loc=4)
plt.show()
