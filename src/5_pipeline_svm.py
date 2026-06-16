# =============================================================================
# ETAPA 3 — PRÉ-PROCESSAMENTO, MODELAGEM E AVALIAÇÃO
# Projeto: Predição de Doença Cardiovascular
# Dataset: Cardiovascular Disease Dataset (Cardio Train) — Kaggle
# Algoritmo: Support Vector Machine (SVM)
# Métrica principal: Recall (contexto clínico — minimizar falsos negativos)
# =============================================================================
# INSTRUÇÕES:
#   1. Certifique-se de ter o arquivo 'cardio_train_sem_valores_invalidos.csv' em src/
#   2. Execute a partir da raiz: python src/5_pipeline_svm.py
#   3. Gráficos salvos em 'graficos/' e resultados em 'graficos/resultados_svm.csv'
# =============================================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

# =============================================================================
# 1. CARREGAMENTO DOS DADOS
# =============================================================================
print("=" * 65)
print("  ETAPA 3 — PRÉ-PROCESSAMENTO E MODELAGEM")
print("  Predição de Doença Cardiovascular — SVM")
print("=" * 65)

print("\n[1/5] Carregando o dataset...")

try:
    df = pd.read_csv("cardio_train_sem_valores_invalidos.csv", sep=";")
    df.columns = df.columns.str.strip()
    print(f"  ✓ Dataset carregado: {df.shape[0]:,} registros × {df.shape[1]} colunas")
except FileNotFoundError:
    print("  ✗ ERRO: 'cardio_train_sem_valores_invalidos.csv' não encontrado na raiz do projeto.")
    exit()

n_original = len(df)

# Remove colunas derivadas pré-existentes no CSV para evitar duplicidade ao recriá-las
for col in ['age_years', 'IMC', 'bmi']:
    if col in df.columns:
        df = df.drop(columns=[col])

# =============================================================================
# 2. PRÉ-PROCESSAMENTO (idêntico ao pipeline Random Forest)
# =============================================================================
print("\n[2/5] Pré-processamento dos dados...")
print("-" * 40)

# --- 2.1 Remoção do ID ---
df = df.drop(columns=['id'])
print("  ✓ Coluna 'id' removida (não é feature preditiva).")

# --- 2.2 Conversão de idade: dias → anos ---
df['age'] = (df['age'] / 365.25).round(1)
df.rename(columns={'age': 'age_years'}, inplace=True)
print("  ✓ Idade convertida de dias para anos (age_years = age / 365.25).")

# --- 2.3 Engenharia de feature: IMC ---
df['IMC'] = (df['weight'] / ((df['height'] / 100) ** 2)).round(2)
print("  ✓ IMC calculado (IMC = weight / (height/100)²).")

# --- 2.4 Remoção de registros clinicamente inválidos ---
mask_invalidos = (
    (df['ap_hi'] <= 0)   | (df['ap_hi'] > 300)  |
    (df['ap_lo'] <= 0)   | (df['ap_lo'] > 200)  |
    (df['ap_hi'] < df['ap_lo'])                  |
    (df['height'] < 100) | (df['height'] > 220)  |
    (df['weight'] < 30)  | (df['weight'] > 200)
)

n_invalidos = mask_invalidos.sum()
df_clean = df[~mask_invalidos].reset_index(drop=True)

print(f"\n  LIMPEZA DE REGISTROS INVÁLIDOS:")
print(f"  ├── Registros originais     : {n_original:,}")
print(f"  ├── Registros removidos     : {n_invalidos:,} ({n_invalidos/n_original*100:.2f}%)")
print(f"  └── Registros após limpeza  : {len(df_clean):,}")

# =============================================================================
# 3. SEPARAÇÃO FEATURES / ALVO E DIVISÃO TREINO/TESTE
# =============================================================================
print("\n[3/5] Separando features e dividindo os dados...")

continuous_cols  = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'IMC']
passthrough_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

X = df_clean[continuous_cols + passthrough_cols]
y = df_clean['cardio']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  ├── Treino : {len(X_train):,} registros (80%) — estratificado")
print(f"  └── Teste  : {len(X_test):,} registros (20%) — estratificado")

# =============================================================================
# 4. NORMALIZAÇÃO (apenas colunas contínuas)
# =============================================================================
print("\n[4/5] Normalizando features contínuas...")

preprocessor = ColumnTransformer(transformers=[
    ('scaler',      StandardScaler(), continuous_cols),
    ('passthrough', 'passthrough',    passthrough_cols)
], remainder='drop')

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)
print("  ✓ StandardScaler aplicado em colunas contínuas.")

os.makedirs("graficos", exist_ok=True)

# =============================================================================
# 5. MODELOS SVM
# =============================================================================
print("\n[5/5] Treinando e avaliando modelos SVM...")

modelos = {
    "SVM_RBF_Padrao": SVC(kernel="rbf", C=1.0, gamma="scale"),
    "SVM_RBF_C10": SVC(kernel="rbf", C=10, gamma="scale"),
    "SVM_RBF_Gamma01": SVC(kernel="rbf", C=1.0, gamma=0.1),
    "SVM_Linear": SVC(kernel="linear", C=1.0),
    "SVM_Poly_Grau3": SVC(kernel="poly", degree=3, C=1.0, gamma="scale"),
    "SVM_Balanceado": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
}

resultados = []

for nome, modelo in modelos.items():
    print("\n" + "=" * 60)
    print(nome)
    print("=" * 60)

    modelo.fit(X_train_proc, y_train)
    y_pred = modelo.predict(X_test_proc)
    scores = modelo.decision_function(X_test_proc)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, scores)

    resultados.append({
        "Modelo":   nome,
        "Recall":   rec,
        "Precisao": prec,
        "F1":       f1,
        "AUC_ROC":  auc,
        "Acuracia": acc
    })

    print(f"\n  Recall (métrica principal): {rec:.4f}")
    print(f"  AUC-ROC                  : {auc:.4f}")
    print(f"  Acurácia                 : {acc:.4f}")
    print(f"  Precisão                 : {prec:.4f}")
    print(f"  F1-Score                 : {f1:.4f}")

    print("\nMatriz de Confusão:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred,
                                target_names=['Sem doença', 'Com doença']))

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Sem doença', 'Com doença'],
                yticklabels=['Sem doença', 'Com doença'])
    plt.title(f"Matriz de Confusão - {nome}")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(f"graficos/matriz_{nome}.png")
    plt.close()

# =============================================================================
# RESULTADOS E RANKINGS
# =============================================================================
df_resultados = pd.DataFrame(resultados)

# --- RANKING PRINCIPAL: RECALL ---
print("\n\n" + "=" * 70)
print("RANKING POR RECALL (métrica principal — contexto clínico)")
print("=" * 70)
ranking_recall = df_resultados.sort_values(by="Recall", ascending=False)
print(ranking_recall[['Modelo', 'Recall', 'Precisao', 'F1', 'AUC_ROC', 'Acuracia']].to_string(index=False))

# --- Rankings complementares ---
print("\n\n" + "=" * 70)
print("RANKING POR AUC-ROC")
print("=" * 70)
ranking_auc = df_resultados.sort_values(by="AUC_ROC", ascending=False)
print(ranking_auc[['Modelo', 'AUC_ROC', 'Recall', 'Acuracia']].to_string(index=False))

print("\n\n" + "=" * 70)
print("RANKING POR ACURÁCIA")
print("=" * 70)
ranking_acc = df_resultados.sort_values(by="Acuracia", ascending=False)
print(ranking_acc[['Modelo', 'Acuracia', 'Recall', 'AUC_ROC']].to_string(index=False))

# --- Salvar resultados ---
df_resultados.to_csv("graficos/resultados_svm.csv", index=False)

# =============================================================================
# GRÁFICOS COMPARATIVOS
# =============================================================================

# Recall — principal
plt.figure(figsize=(10, 6))
sns.barplot(data=ranking_recall, x="Modelo", y="Recall")
plt.title("Comparação dos Modelos SVM - Recall (Métrica Principal)")
plt.xlabel("Modelo")
plt.ylabel("Recall")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("graficos/comparacao_svm_recall.png")
plt.close()

# Acurácia
plt.figure(figsize=(10, 6))
sns.barplot(data=ranking_acc, x="Modelo", y="Acuracia")
plt.title("Comparação dos Modelos SVM - Acurácia")
plt.xlabel("Modelo")
plt.ylabel("Acurácia")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("graficos/comparacao_svm_acuracia.png")
plt.close()

# AUC-ROC
plt.figure(figsize=(10, 6))
sns.barplot(data=ranking_auc, x="Modelo", y="AUC_ROC")
plt.title("Comparação dos Modelos SVM - AUC-ROC")
plt.xlabel("Modelo")
plt.ylabel("AUC-ROC")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("graficos/comparacao_svm_auc.png")
plt.close()

# =============================================================================
# MELHOR MODELO
# =============================================================================
melhor_recall = ranking_recall.iloc[0]

print("\n\n" + "=" * 70)
print("MELHOR MODELO POR RECALL (critério principal)")
print("=" * 70)
print(f"  Modelo   : {melhor_recall['Modelo']}")
print(f"  Recall   : {melhor_recall['Recall']:.4f}")
print(f"  AUC-ROC  : {melhor_recall['AUC_ROC']:.4f}")
print(f"  Acurácia : {melhor_recall['Acuracia']:.4f}")
print(f"  Precisão : {melhor_recall['Precisao']:.4f}")
print(f"  F1-Score : {melhor_recall['F1']:.4f}")

print("\nPipeline executado com sucesso!")
