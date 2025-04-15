# 1. Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Загрузка данных
df = pd.read_csv("Student_Mental_Stress_and_Coping_Mechanisms.csv")

# 3. afisarea
df_copy = df
dft = df
df_corr = df
df_copy

# 4. afisarea informatiei
print("Информация о датафрейме:")
df.info()

# 5. Очистка названий колонок
df.columns = df.columns.str.strip()

# 6. Проверка пропущенных значений
print("\nПропущенные значения:\n", df.isnull().sum())

# 7. Кодировка категориальных признаков
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col].astype(str))

# 8. Целевая переменная — бинаризация по уровню стресса
df["StressBinary"] = df["Mental Stress Level"].apply(lambda x: 1 if x >= 2 else 0)

# 9. Корреляции и выбор топ-3 признаков
corr_matrix = df.corr()
target_corr = (
    corr_matrix["StressBinary"].drop("StressBinary").sort_values(ascending=False)
)
top_features = target_corr.head(3).index.tolist()

print("\n🧠 Топ-3 признака по корреляции с уровнем стресса:")
print(target_corr.head(3))

# 10. Подготовка данных
X = df[top_features]
y = df["StressBinary"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 11. Обучение моделей
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n📊 Модель: {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Предсказано")
    plt.ylabel("Фактическое")
    plt.tight_layout()
    plt.show()

# 12. Визуализация целевой переменной
plt.figure(figsize=(6, 4))
sns.countplot(x=y, palette="coolwarm")
plt.title("Распределение уровня стресса (бинарно)")
plt.xlabel("Стресс: 0 = низкий, 1 = высокий")
plt.ylabel("Количество")
plt.tight_layout()
plt.show()

# 13. Тепловая карта корреляций
plt.figure(figsize=(14, 10))
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5
)
plt.title("Корреляционная матрица всех признаков")
plt.tight_layout()
plt.show()
