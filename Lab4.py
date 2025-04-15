import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# Încărcarea datelor
df = pd.read_csv("wine-quality-white-and-red.csv")

# Verificarea valorilor lipsă
print(df.isna().sum())

# describe() pentru a obține statistici descriptive
print(df.describe())

# Conversia tipului de vin în numeric
df["type"] = df["type"].map({"red": 0, "white": 1})

# Группировка по типу вина и расчет статистик для каждой группы, var это мера разброса данных, показывающая, насколько значения отклоняются от среднего
statistical_analysis = df.groupby("type").agg(
    [
        "mean",
        "median",
        "std",
        "var",
        lambda x: x.max() - x.min(),
    ]  # Медиана — это центральное значение для высоких и низких значений, std — это стандартное отклонение, var — это дисперсия, а lambda x: x.max() - x.min() — это разница между максимальным и минимальным значением
)
print(statistical_analysis)

# Анализа кореляции между признаками и типом вина
correlation = df.corr()["type"].abs().sort_values(ascending=False)
print("Corelația cu tipul de vin:\n", correlation)

# вИзуализация 1: Гистограмма летучей кислотности по типам вин
sns.histplot(data=df, x="volatile acidity", hue="type")
plt.show()

# Визуализация 2: Боксплот содержания алкоголя по типам вин
sns.boxplot(data=df, x="type", y="alcohol")
plt.show()

# Corectare heatmap: rotire etichete + layout ajustat
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Matricea corelațiilor")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Alegerea celor mai corelate 3 caracteristici
top_features = ["total sulfur dioxide", "volatile acidity", "chlorides"]
X = df[top_features]
y = df["type"]

# Împărțirea datelor în set de antrenare și testare
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Antrenarea modelului Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Predicții și evaluare
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Red", "White"])

print(f"Acuratețea modelului: {accuracy:.2%}")
print("Raportul de clasificare:\n", report)

# Vizualizare distribuții pentru cele 3 caracteristici
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, feature in enumerate(top_features):
    sns.histplot(
        data=df,
        x=feature,
        hue="type",
        kde=True,
        ax=axes[i],
        palette="Set1",
        element="step",
        common_norm=False,
    )
    axes[i].set_title(f"Distribuția caracteristicii: {feature}")

plt.tight_layout()
plt.show()
