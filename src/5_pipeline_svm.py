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
    confusion_matrix,
    roc_auc_score
)

# ====================================
# CARREGAR DATASET
# ====================================

df = pd.read_csv(
    "cardio_train_sem_valores_invalidos.csv",
    sep=";"
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
# PASTA DE SAÍDA
# ====================================

os.makedirs("graficos", exist_ok=True)

# ====================================
# MODELOS SVM
# ====================================

modelos = {

    "SVM_RBF_Padrao": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    ),

    "SVM_RBF_C10": SVC(
        kernel="rbf",
        C=10,
        gamma="scale"
    ),

    "SVM_RBF_Gamma01": SVC(
        kernel="rbf",
        C=1.0,
        gamma=0.1
    ),

    "SVM_Linear": SVC(
        kernel="linear",
        C=1.0
    ),

    "SVM_Poly_Grau3": SVC(
        kernel="poly",
        degree=3,
        C=1.0,
        gamma="scale"
    ),

    "SVM_Balanceado": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced"
    )
}

# ====================================
# EXECUTAR MODELOS
# ====================================

resultados = []

for nome, modelo in modelos.items():

    print("\n" + "=" * 60)
    print(nome)
    print("=" * 60)

    # Treinamento
    modelo.fit(X_train, y_train)

    # Predições
    y_pred = modelo.predict(X_test)

    # Scores para cálculo da AUC
    scores = modelo.decision_function(X_test)

    # Métricas
    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(
        y_test,
        scores
    )

    resultados.append({
        "Modelo": nome,
        "Acuracia": accuracy,
        "AUC_ROC": auc
    })

    print(f"\nAcurácia: {accuracy:.4f}")
    print(f"AUC-ROC:  {auc:.4f}")

    print("\nMatriz de Confusão:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\nRelatório de Classificação:")
    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    # ====================================
    # SALVAR MATRIZ DE CONFUSÃO
    # ====================================

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(f"Matriz de Confusão - {nome}")
    plt.xlabel("Previsto")
    plt.ylabel("Real")

    plt.tight_layout()

    plt.savefig(
        f"graficos/matriz_{nome}.png"
    )

    plt.close()

# ====================================
# RESULTADOS
# ====================================

df_resultados = pd.DataFrame(resultados)

# ====================================
# RANKING POR ACURÁCIA
# ====================================

print("\n")
print("=" * 70)
print("RANKING POR ACURÁCIA")
print("=" * 70)

ranking_acuracia = df_resultados.sort_values(
    by="Acuracia",
    ascending=False
)

print(ranking_acuracia)

# ====================================
# RANKING POR AUC-ROC
# ====================================

print("\n")
print("=" * 70)
print("RANKING POR AUC-ROC")
print("=" * 70)

ranking_auc = df_resultados.sort_values(
    by="AUC_ROC",
    ascending=False
)

print(ranking_auc)

# ====================================
# SALVAR RESULTADOS
# ====================================

df_resultados.to_csv(
    "graficos/resultados_svm.csv",
    index=False
)

# ====================================
# GRÁFICO DE ACURÁCIA
# ====================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=ranking_acuracia,
    x="Modelo",
    y="Acuracia"
)

plt.title("Comparação dos Modelos SVM - Acurácia")
plt.xlabel("Modelo")
plt.ylabel("Acurácia")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "graficos/comparacao_svm_acuracia.png"
)

plt.close()

# ====================================
# GRÁFICO DE AUC-ROC
# ====================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=ranking_auc,
    x="Modelo",
    y="AUC_ROC"
)

plt.title("Comparação dos Modelos SVM - AUC-ROC")
plt.xlabel("Modelo")
plt.ylabel("AUC-ROC")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "graficos/comparacao_svm_auc.png"
)

plt.close()

# ====================================
# MELHOR MODELO POR ACURÁCIA
# ====================================

melhor_acc = ranking_acuracia.iloc[0]

print("\n")
print("=" * 70)
print("MELHOR MODELO POR ACURÁCIA")
print("=" * 70)

print(f"Modelo: {melhor_acc['Modelo']}")
print(f"Acurácia: {melhor_acc['Acuracia']:.4f}")
print(f"AUC-ROC:  {melhor_acc['AUC_ROC']:.4f}")

# ====================================
# MELHOR MODELO POR AUC
# ====================================

melhor_auc = ranking_auc.iloc[0]

print("\n")
print("=" * 70)
print("MELHOR MODELO POR AUC-ROC")
print("=" * 70)

print(f"Modelo: {melhor_auc['Modelo']}")
print(f"Acurácia: {melhor_auc['Acuracia']:.4f}")
print(f"AUC-ROC:  {melhor_auc['AUC_ROC']:.4f}")

print("\nPipeline executado com sucesso!")