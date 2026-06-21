# =============================================================================
# ETAPA 3 — PRÉ-PROCESSAMENTO, MODELAGEM E AVALIAÇÃO
# Projeto: Predição de Doença Cardiovascular
# Dataset: Cardiovascular Disease Dataset (Cardio Train) — Kaggle
# Algoritmo: Regressão Logística
# Métrica principal: Recall (contexto clínico — minimizar falsos negativos)
# =============================================================================
# INSTRUÇÕES:
#   1. Certifique-se de ter o arquivo 'cardio_train_sem_valores_invalidos.csv' em src/
#   2. Execute a partir da raiz: python src/6_pipeline_logistic_regression.py
#   3. Gráficos salvos em 'graficos/' com prefixo 'lr_'
# =============================================================================

import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_theme(style="whitegrid", palette="muted")

os.makedirs('graficos', exist_ok=True)

print("=" * 65)
print("  ETAPA 3 — PRÉ-PROCESSAMENTO E MODELAGEM")
print("  Predição de Doença Cardiovascular — Regressão Logística")
print("=" * 65)


# =============================================================================
# 1. CARREGAMENTO DOS DADOS
# =============================================================================
print("\n[1/6] Carregando o dataset...")

try:
    df = pd.read_csv("cardio_train_sem_valores_invalidos.csv", sep=';')
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
# 2. PRÉ-PROCESSAMENTO
# =============================================================================
print("\n[2/6] Pré-processamento dos dados...")
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
print("\n[3/6] Separando features e dividindo os dados...")

X = df_clean.drop(columns=['cardio'])
y = df_clean['cardio']

continuous_cols  = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'IMC']
passthrough_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  ├── Treino : {len(X_train):,} registros (80%) — estratificado")
print(f"  └── Teste  : {len(X_test):,} registros (20%) — estratificado")


# =============================================================================
# 4. CONSTRUÇÃO DO PIPELINE E TREINAMENTO
# =============================================================================
print("\n[4/6] Construindo e treinando o pipeline...")
print("-" * 40)

preprocessor = ColumnTransformer(transformers=[
    ('scaler',      StandardScaler(), continuous_cols),
    ('passthrough', 'passthrough',    passthrough_cols)
], remainder='drop')

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='lbfgs',
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model',        model)
])

print("  Treinando Regressão Logística...")
pipeline.fit(X_train, y_train)
print("  ✓ Pipeline treinado com sucesso.")


# =============================================================================
# 5. AVALIAÇÃO DO MODELO
# =============================================================================
print("\n[5/6] Avaliando o modelo...")
print("-" * 40)

y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
auc_roc   = roc_auc_score(y_test, y_proba)

print(f"""
  MÉTRICAS DE DESEMPENHO — conjunto de teste:
  ┌─────────────────────────────────────────────┐
  │  Recall   ★ (métrica principal)  :  {recall:.4f}  │
  │  AUC-ROC                         :  {auc_roc:.4f}  │
  │  Acurácia                        :  {accuracy:.4f}  │
  │  Precisão                        :  {precision:.4f}  │
  │  F1-Score                        :  {f1:.4f}  │
  └─────────────────────────────────────────────┘
""")

print("RELATÓRIO DE CLASSIFICAÇÃO POR CLASSE:")
print(classification_report(y_test, y_pred,
                            target_names=['Sem doença (0)', 'Com doença (1)']))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"  MATRIZ DE CONFUSÃO:")
print(f"  ├── TN (Verdadeiros Negativos)  : {tn:,}  — sem doença, predito corretamente")
print(f"  ├── FP (Falsos Positivos)       : {fp:,}  — sem doença, predito como doente")
print(f"  ├── FN (Falsos Negativos)       : {fn:,}  — doente, predito como saudável ⚠")
print(f"  └── TP (Verdadeiros Positivos)  : {tp:,}  — doente, predito corretamente")

print(f"\n  VALIDAÇÃO CRUZADA ESTRATIFICADA (5-fold) — AUC-ROC:")
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
print(f"  Média : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Coeficientes do modelo
print(f"\n  COEFICIENTES DO MODELO (interpretabilidade):")
feature_names = continuous_cols + passthrough_cols
coef_df = pd.DataFrame({
    'feature': feature_names,
    'coeficiente': pipeline.named_steps['model'].coef_[0]
}).sort_values('coeficiente', key=abs, ascending=False)
print(coef_df.to_string(index=False))


# =============================================================================
# 6. VISUALIZAÇÕES
# =============================================================================
print("\n[6/6] Gerando visualizações...")

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(9, 8))
ax.plot(fpr, tpr, color='#4C72B0', lw=2.5,
        label=f'Logistic Regression (AUC = {auc_roc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Classificador aleatório (AUC = 0.50)')
ax.fill_between(fpr, tpr, alpha=0.08, color='#4C72B0')
ax.set_xlabel('Taxa de Falsos Positivos (1 − Especificidade)', fontsize=12)
ax.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=12)
ax.set_title('Curva ROC — Regressão Logística\nPredição de Doença Cardiovascular',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graficos/lr_curva_roc.png', bbox_inches='tight')
plt.close()

# Matriz de confusão
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Sem doença', 'Com doença'],
            yticklabels=['Sem doença', 'Com doença'],
            annot_kws={'size': 14, 'weight': 'bold'})
ax.set_xlabel('Predito', fontsize=12)
ax.set_ylabel('Real', fontsize=12)
ax.set_title('Matriz de Confusão — Regressão Logística', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('graficos/lr_matriz_confusao.png', bbox_inches='tight')
plt.close()

# Coeficientes
coef_plot = coef_df.sort_values('coeficiente', ascending=True).reset_index(drop=True)
colors = ['#4C72B0' if x < 0 else '#C44E52' for x in coef_plot['coeficiente']]
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(coef_plot['feature'], coef_plot['coeficiente'], color=colors, edgecolor='white')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Coeficiente (log-odds)', fontsize=11)
ax.set_title('Coeficientes do Modelo — Regressão Logística\n(Impacto na probabilidade de doença)',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.legend(handles=[
    plt.Rectangle((0, 0), 1, 1, color='#C44E52', label='Aumenta risco (coef. +)'),
    plt.Rectangle((0, 0), 1, 1, color='#4C72B0', label='Diminui risco (coef. -)')
], fontsize=10, loc='lower right')
plt.tight_layout()
plt.savefig('graficos/lr_coeficientes.png', bbox_inches='tight')
plt.close()

print("  ✓ Gráficos salvos com sucesso na pasta 'graficos/'.")

print("\n" + "=" * 65)
print("  Etapa 3 — Regressão Logística concluída com sucesso!")
print("=" * 65)
