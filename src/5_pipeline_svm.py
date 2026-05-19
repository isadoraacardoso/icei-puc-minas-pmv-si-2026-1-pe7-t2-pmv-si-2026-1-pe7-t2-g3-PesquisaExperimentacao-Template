import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ====================================
# CARREGAR DATASET
# ====================================
# ====================================
# CARREGAR DATASET
# ====================================

df = pd.read_csv(
    "cardio_train_sem_valores_invalidos.csv",
    sep=';'
)

df.columns = df.columns.str.strip()

# ====================================
# FEATURES E TARGET
# ====================================

X = df.drop("cardio", axis=1)
y = df["cardio"]

# ====================================
# DIVISÃO TREINO / TESTE
# ====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ====================================
# NORMALIZAÇÃO
# ====================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ====================================
# MODELO SVM
# ====================================

modelo = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale'
)

# ====================================
# TREINAMENTO
# ====================================

modelo.fit(X_train, y_train)

# ====================================
# PREVISÕES
# ====================================

y_pred = modelo.predict(X_test)

# ====================================
# MÉTRICAS
# ====================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("RESULTADOS - SVM")
print("==============================")

print(f"\nAcurácia: {accuracy:.4f}")

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# ====================================
# CRIAR PASTA DE GRÁFICOS
# ====================================

os.makedirs("graficos", exist_ok=True)

# ====================================
# MATRIZ DE CONFUSÃO VISUAL
# ====================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Matriz de Confusão - SVM")
plt.xlabel("Previsto")
plt.ylabel("Real")

plt.tight_layout()

plt.savefig("graficos/matriz_confusao_svm.png")

plt.show()

# ====================================
# SALVAR RELATÓRIO
# ====================================

with open("graficos/relatorio_svm.txt", "w", encoding="utf-8") as f:

    f.write("RESULTADOS - SVM\n")
    f.write("========================\n\n")

    f.write(f"Acurácia: {accuracy:.4f}\n\n")

    f.write("Matriz de Confusão:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))

    f.write("\n\nRelatório de Classificação:\n")
    f.write(classification_report(y_test, y_pred))

print("\nPipeline SVM executado com sucesso!")