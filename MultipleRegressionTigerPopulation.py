import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data_df= pd.read_csv('my_data.csv')
X = data_df.drop(['TigerPopulation'],axis=1).values
Y = data_df['TigerPopulation'].values

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.3,random_state=0)

model = LinearRegression()
model.fit(X_train,Y_train)
predicted_Y_values = model.predict(X_test)

## MODEL EVALUATION
print(mean_squared_error(Y_test,predicted_Y_values))
print(r2_score(Y_test,predicted_Y_values))

##PREDICTION
print(model.predict([[2012,5,12,25000,120,2]]))

##PLOT
plt.figure(figsize=(15,10))
plt.scatter(Y_test,predicted_Y_values)
plt.xlabel('ACTUAL')
plt.ylabel('PREDICTED')
plt.title('ACTUAL vs Predicted')
plt.show()
