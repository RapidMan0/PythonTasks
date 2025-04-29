# Импорт библиотек
import pandas as pd
import numpy as np
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

# Просмотр первых строк
print("Предпросмотр данных:")
print(df.head())

print (df.info())
# Предобработка числовых данных
df["Amount"] = df["Amount"].replace("[\$,]", "", regex=True).astype(float)

# Целевая переменная — 'Product'
target_column = "Product" # Целевая переменная тип шоколада

# Разделение на признаки (X) и целевую переменную (y)
X = df.drop(columns=[target_column, "Date"])  # Исключаем целевой признак и дату
y = df[target_column] 

# Преобразование категориальных признаков в числовые (если есть)
X = pd.get_dummies(X, drop_first=True) # Категориальные переменные (например, регион, месяц) преобразуются в числовой формат с помощью pd.get_dummies().

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
best_accuracy = accuracy_scores[best_model_name]

print("\nРезультаты моделей:")
for name, acc in accuracy_scores.items():
    print(f"{name}: {acc:.4f}")

print(f"\nЛучшая модель: {best_model_name} с точностью: {best_accuracy:.4f}")

# Step 7: Conclusion
print("\nConclusion:")
print(
    f"After training and evaluating five machine learning models (KNN, Decision Tree, Random Forest, "
    f"Logistic Regression, and Naive Bayes) on the Chocolate Sales Dataset, the model with the "
    f"highest accuracy on the test subset is {best_model_name} with an accuracy of {best_accuracy:.4f}. "
    f"This model is the most effective among those tested for predicting the type of chocolate product "
    f"based on sales-related features such as region, month, unit price, and quantity."
)
