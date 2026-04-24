import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

df = pd.read_csv("assets/energy_data.csv")

X = df[["temperature", "humidity", "hour", "is_weekend"]]
y = df["consumption"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

error = mean_absolute_percentage_error(y_test, y_pred) * 100
print(f"Середня помилка: {error:.2f}%")

plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')

plt.xlabel("Справжнє споживання")
plt.ylabel("Прогнозоване")
plt.title("Справжнє vs Прогнозоване")

plt.grid()
plt.show()