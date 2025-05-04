# Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# Загрузка датасета
df = pd.read_csv("ChocolateSales.csv")

# Преобразование даты в datetime и извлечение месяца
df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%y")
df["Month"] = df["Date"].dt.month_name()

# Очистка и преобразование суммы продажи
df["Amount"] = df["Amount"].replace("[\$,]", "", regex=True).astype(float)

# Фильтрация по стране India
df_india = df[df["Country"] == "India"]

# Группировка по месяцу и продукту с суммированием продаж
monthly_sales = df_india.groupby(["Month", "Product"])["Amount"].sum().reset_index()

# Переупорядочим месяцы
month_order = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
monthly_sales["Month"] = pd.Categorical(
    monthly_sales["Month"], categories=month_order, ordered=True
)
monthly_sales = monthly_sales.sort_values("Month")

# Построим график продаж
# Построим график продаж с уникальными цветами
plt.figure(figsize=(12, 6))

# Создаем список уникальных цветов
colors = plt.cm.tab20.colors  # Используем палитру из 20 цветов
product_colors = {
    product: colors[i % len(colors)]
    for i, product in enumerate(monthly_sales["Product"].unique())
}

for product in monthly_sales["Product"].unique():
    data = monthly_sales[monthly_sales["Product"] == product]
    plt.plot(
        data["Month"],
        data["Amount"],
        marker="o",
        label=product,
        color=product_colors[product],
    )

plt.title("Продажи шоколадных продуктов по месяцам (India)")
plt.xlabel("Месяц")
plt.ylabel("Сумма продаж ($)")
plt.legend(title="Продукт", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

# Подготовка данных для моделей
X = df_india[["Month", "Amount", "Boxes Shipped"]]
X["Month"] = pd.Categorical(
    X["Month"], categories=month_order, ordered=True
).codes  # Преобразуем месяцы в числовой формат
y = df_india["Product"]

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Список моделей
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": GaussianNB(),
}

# Обучение и оценка точности моделей
accuracy_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_scores[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# Выбор лучшей модели
best_model_name = max(accuracy_scores, key=accuracy_scores.get)
best_model = models[best_model_name]
print(
    f"\nЛучшая модель: {best_model_name} с точностью: {accuracy_scores[best_model_name]:.4f}"
)
