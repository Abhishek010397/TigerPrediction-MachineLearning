###MULTIPLE REGRESSION
###Multiple regression is like linear regression, but with more than one independent value, meaning that we try to predict a value based on two or more variables.

import pandas
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
df = pandas.read_csv("my_data.csv")

X = df[['YEAR', 'Poaching','HabitatLoss','PreyPopulation','PreyPoaching','RetaliationKilling']]
y = df['TigerPopulation']

# z = (x - u) / s Where z is the new value, x is the original value, u is the mean and s is the standard deviation.
#Python sklearn module has a method called StandardScaler() which returns a Scaler object with methods for transforming data sets.

scaledX = scale.fit_transform(X)
regr = linear_model.LinearRegression()
regr.fit(scaledX, y)
scaled = scale.transform([[2021,6,10.16,40000,120,0]])
predictedTigerPopulation = regr.predict([scaled[0]])
print(int(predictedTigerPopulation))
