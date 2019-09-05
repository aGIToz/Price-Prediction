from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pandas_datareader.data as web
from pandas import Series, DataFrame
import config as cf
import datetime
import operator
import math

# create the dataframe
df = web.DataReader(cf.company, 'yahoo', cf.start, cf.end)

#create a dataframe for stock price prediction
dfreg = df.loc[:,['Adj Close','Volume']]
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

#PRE-PROCESS 
# drop missing value rows
dfreg.dropna(inplace=True)

# predict the stock price for the next 18 days
forecast_out = int(cf.days)

# separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))

# Scale the X so that everyone can have the same distribution for linear regression
scaler =  preprocessing.StandardScaler()
X = scaler.fit_transform(X)

# create the prediction set
X_lately= X[-forecast_out:]
X= X[:-forecast_out]

# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# MODEL GENERATION
# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)

# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)

# Bayesian Regression
clfbay = BayesianRidge()
clfbay.fit(X_train, y_train)

#get the accuracy of the models
confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test,y_test)
confidencepoly3 = clfpoly3.score(X_test,y_test)
confidenceknn = clfknn.score(X_test, y_test)
confidencebay = clfbay.score(X_test,y_test)

try:
	dict1 = {'simple linear regression':confidencereg, 'polynomial2 regression':confidencepoly2, 'poly3 regression':confidencepoly3,  'knn':confidenceknn, 'bayesian regression':confidencebay}
	dict2 = {'simple linear regression':clfreg, 'polynomial2 regression':clfpoly2, 'poly3 regression':clfpoly3,  'knn':clfknn, 'bayesian regression':clfbay}
	best = max(dict1.items(), key=operator.itemgetter(1))[0]
	print("Out of all the models the  best model is:", best, "with the accuracy of:" ,np.around(dict1[best],3))

except: 
	print("Not able to find the max value in the dictionary, using the code as suggested here https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary for python3")

else:

	pass

print("Forecasting and ploting for the 18 days...")
# forecaste for the next 18 days
forecast_set = dict2[best].predict(X_lately)
dfreg['Forecast'] = np.nan
print("The prediction for the next ", cf.days, "are: ", forecast_set)

#plot 
last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
dfreg['Adj Close'].tail(250).plot()
dfreg['Forecast'].tail(250).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
