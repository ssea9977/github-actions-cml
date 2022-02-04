from datetime import datetime 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import pickle

input_file = "/home/ocean/mlops/taxi-price/yellow_tripdata_2021-01.csv"

df = pd.read_csv(input_file, header=0, low_memory=False)
df = df.loc[(df["total_amount"] < 20) & (df["trip_distance"] < 100)]
distance = df["trip_distance"]
hours = df["tpep_pickup_datetime"].map(
  lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour
)
data = pd.concat([hours, distance], axis=1)

target = df["total_amount"]

train_x, test_x, train_y, test_y = train_test_split(
  data,
  target,
  test_size = 0.2,
  random_state = 24
)

lr = LinearRegression()
lr.fit(train_x, train_y)

prediction = lr.predict(test_x)
train_prediction = lr.predict(train_x)

mse = metrics.mean_squared_error(test_y, prediction)
r2 = r2_score(test_y, prediction)
print(mse)
print(r2)
print("PREDICTION")
print(prediction[:5])
print("LABEL")
print(test_y[:5])

with open("metrics.txt", 'w') as outfile:
        outfile.write("mse: %2.1f\n" % mse)
        outfile.write("r2: %2.1f\n" % r2)

plt.scatter(prediction[:30], prediction[:30]-train_y[:30], c='blue', marker='o', label='training data')
plt.scatter(train_prediction[:30], train_prediction[:30]-test_y[:30], c='lightgreen', marker='o', label='test data')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlabel('prediction')
plt.ylabel('residual')
plt.title('yellow-taxi-price-prediction')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("line.png",dpi=120) 

# with open("model.pkl", "wb") as f:
#   pickle.dump(lr, f)