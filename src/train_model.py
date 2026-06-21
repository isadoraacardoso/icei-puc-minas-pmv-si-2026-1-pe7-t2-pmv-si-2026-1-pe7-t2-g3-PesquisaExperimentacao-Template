# =============================================================================
# SCRIPT DE TREINAMENTO E EXPORTAÇÃO DO MODELO
# Predição de Doença Cardiovascular — Random Forest
# =============================================================================
# Este script treina o modelo Random Forest e o exporta em formato pickle (.pkl)
# Uso: python train_model.py
# =============================================================================

import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

warnings.filterwarnings('ignore')

print("=" * 70)
print("  TREINAMENTO E EXPORTAÇÃO DO MODELO — RANDOM FOREST")
print("  Predição de Doença Cardiovascular")
print("=" * 70)


# =============================================================================
# 1. CARREGAMENTO DOS DADOS
# =============================================================================
print("\n[1/5] Carregando dados...")

try:
    df = pd.read_csv("cardio_train_sem_valores_invalidos.csv", sep=';')
    print(f"  ✓ Dataset carregado: {df.shape[0]:,} registros × {df.shape[1]} colunas")
except FileNotFoundError:
    print("  ✗ ERRO: 'cardio_train_sem_valores_invalidos.csv' não encontrado.")
    print("     Certifique-se de executar este script a partir da raiz do projeto.")
    exit(1)


# =============================================================================
# 2. PRÉ-PROCESSAMENTO
# =============================================================================
print("\n[2/5] Pré-processando dados...")

# Remover ID
df = df.drop(columns=['id'])

# Converter idade de dias para anos
df['age'] = (df['age'] / 365.25).round(1)
df.rename(columns={'age': 'age_years'}, inplace=True)

# Calcular IMC
df['bmi'] = (df['weight'] / ((df['height'] / 100) ** 2)).round(2)

# Remover registros clinicamente inválidos
mask_invalidos = (
    (df['ap_hi'] <= 0)    | (df['ap_hi'] > 300)   |
    (df['ap_lo'] <= 0)    | (df['ap_lo'] > 200)   |
    (df['ap_hi'] < df['ap_lo'])                    |
    (df['height'] < 100)  | (df['height'] > 220)  |
    (df['weight'] < 30)   | (df['weight'] > 200)
)

df_clean = df[~mask_invalidos].copy()
n_removed = mask_invalidos.sum()
print(f"  ✓ Registros inválidos removidos: {n_removed}")
print(f"  ✓ Dataset após limpeza: {len(df_clean):,} registros")

# Remover duplicatas de colunas e resetar índice
df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()].copy()
df_clean = df_clean.reset_index(drop=True)


# =============================================================================
# 3. FEATURES E ALVO
# =============================================================================
print("\n[3/5] Preparando features e alvo...")

# Separar features e alvo
X = df_clean.drop(columns=['cardio'])
y = df_clean['cardio']

# Definir colunas contínuas e categóricas
continuous_cols = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']
categorical_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

# Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  ✓ Treino: {len(X_train):,} registros (80%)")
print(f"  ✓ Teste:  {len(X_test):,} registros (20%)")


# =============================================================================
# 4. CRIAÇÃO DO PIPELINE E TREINAMENTO
# =============================================================================
print("\n[4/5] Construindo e treinando pipeline...")

# Pré-processador
preprocessor = ColumnTransformer(transformers=[
    ('scaler',      StandardScaler(), continuous_cols),
    ('passthrough', 'passthrough',    categorical_cols)
], remainder='drop')

# Modelo Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1
)

# Pipeline completo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model',        model)
])

print("  Treinando Random Forest (100 árvores)...")
pipeline.fit(X_train, y_train)
print("  ✓ Pipeline treinado com sucesso")


# =============================================================================
# 5. AVALIAÇÃO E EXPORTAÇÃO
# =============================================================================
print("\n[5/5] Avaliando modelo e exportando...")

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_proba)

print(f"""
  MÉTRICAS DE DESEMPENHO (Conjunto de Teste):
  ┌──────────────────────────────────────────┐
  │  AUC-ROC     ★ : {auc_roc:.4f}                 │
  │  Acurácia      : {accuracy:.4f}                 │
  │  Precisão      : {precision:.4f}                 │
  │  Recall        : {recall:.4f}                 │
  │  F1-Score      : {f1:.4f}                 │
  └──────────────────────────────────────────┘
""")

# Exportar pipeline em pickle
model_path = "modelo_rf.pkl"
joblib.dump(pipeline, model_path)
print(f"  ✓ Modelo exportado: '{model_path}'")

# Armazenar informações de feature ordering
feature_info = {
    'continuous_cols': continuous_cols,
    'categorical_cols': categorical_cols,
    'all_cols': continuous_cols + categorical_cols,
    'metrics': {
        'auc_roc': auc_roc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
}

joblib.dump(feature_info, "feature_info.pkl")
print(f"  ✓ Informações de features exportadas: 'feature_info.pkl'")

print("\n" + "=" * 70)
print("  ✓ TREINAMENTO E EXPORTAÇÃO CONCLUÍDOS COM SUCESSO!")
print("=" * 70)