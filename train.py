import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

df = pd.read_csv(r'..\train.csv')

df.dropna(inplace = True)

x = df.iloc[:, [0]]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred  = lr.predict(X_test)

loss = mean_squared_error(y_test, y_pred)

print(f'The loss is: {loss:.3f}')

with open("./model.pkl", "wb") as f:
    pickle.dump(lr, f)

print("yeah buddy!! Model has been saved.")    