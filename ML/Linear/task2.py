import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Читаємо дані
df = pd.read_csv("assets/cars.csv")

# Дані
X = df[["year", "engine_volume", "mileage", "horsepower"]]
y = df["price"]

# Розділення
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Модель
model = LinearRegression()
model.fit(X_train, y_train)

# Прогноз
y_pred = model.predict(X_test)

# 🔥 Рахуємо % помилку
errors = abs((y_test - y_pred) / y_test) * 100

# 🔥 Графік
plt.scatter(X_test["mileage"], errors, alpha=0.6)

plt.xlabel("Пробіг (mileage)")
plt.ylabel("Помилка (%)")
plt.title("Залежність помилки від пробігу")

plt.grid()
plt.show()