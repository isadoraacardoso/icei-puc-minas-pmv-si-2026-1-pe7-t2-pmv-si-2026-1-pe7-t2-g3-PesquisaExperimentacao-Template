# =============================================================================
# ETAPA 3 — PRÉ-PROCESSAMENTO, MODELAGEM E AVALIAÇÃO
# Projeto: Predição de Doença Cardiovascular
# Dataset: Cardiovascular Disease Dataset (Cardio Train) — Kaggle
# Algoritmo: Random Forest
# Métrica principal: AUC-ROC
# =============================================================================
# INSTRUÇÕES:
#   1. Certifique-se de ter o arquivo 'cardio_train_sem_valores_invalidos.csv' na raiz
#   2. Execute a partir da raiz: python src/4_pipeline_etapa3.py
#   3. Os gráficos serão salvos em 'graficos/' e o modelo em 'modelo_cardio.onnx'
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

import os
import warnings
from skl2onnx import to_onnx

warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_theme(style="whitegrid", palette="muted")

os.makedirs('graficos', exist_ok=True)

print("=" * 65)
print("  ETAPA 3 — PRÉ-PROCESSAMENTO E MODELAGEM")
print("  Predição de Doença Cardiovascular — Random Forest")
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
print("  ✓ IMC calculated (IMC = weight / (height/100)²).")

# --- 2.4 Remoção de registros clinicamente inválidos ---
mask_invalidos = (
    (df['ap_hi'] <= 0)    | (df['ap_hi'] > 300)   |   
    (df['ap_lo'] <= 0)    | (df['ap_lo'] > 200)   |   
    (df['ap_hi'] < df['ap_lo'])                    |   
    (df['height'] < 100)  | (df['height'] > 220)  |   
    (df['weight'] < 30)   | (df['weight'] > 200)      
)

n_invalidos = mask_invalidos.sum()
df_clean = df[~mask_invalidos].copy()

print(f"\n  LIMPEZA DE REGISTROS INVÁLIDOS:")
print(f"  ├── Registros originais     : {n_original:,}")
print(f"  ├── Registros removidos     : {n_invalidos:,} ({n_invalidos/n_original*100:.2f}%)")
print(f"  └── Registros após limpeza  : {len(df_clean):,}")

# --- 2.5 Correção de colunas duplicadas e reset de índice ---
df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()].copy()
df_clean = df_clean.reset_index(drop=True)


# =============================================================================
# 3. SEPARAÇÃO FEATURES / ALVO E DIVISÃO TREINO/TESTE
# =============================================================================
print("\n[3/6] Separando features e dividindo os dados...")

X = df_clean.drop(columns=['cardio'])
y = df_clean['cardio']

continuous_cols = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'IMC']
passthrough_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  ├── Treino : {len(X_train):,} registros (80%)")
print(f"  └── Teste  : {len(X_test):,} registros (20%)")


# =============================================================================
# 4. CONSTRUÇÃO DO PIPELINE E TREINAMENTO
# =============================================================================
print("\n[4/6] Construindo e treinando o pipeline...")
print("-" * 40)

preprocessor = ColumnTransformer(transformers=[
    ('scaler',      StandardScaler(), continuous_cols),
    ('passthrough', 'passthrough',    passthrough_cols)
], remainder='drop')

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model',        model)
])

print("  Treinando Random Forest (100 árvores)...")
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
  │  AUC-ROC  ★ (métrica principal)  :  {auc_roc:.4f}  │
  │  Acurácia                        :  {accuracy:.4f}  │
  │  Precisão                        :  {precision:.4f}  │
  │  Recall (Sensibilidade) ★★       :  {recall:.4f}  │
  │  F1-Score                        :  {f1:.4f}  │
  └─────────────────────────────────────────────┘
""")

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n  VALIDAÇÃO CRUZADA ESTRATIFICADA (5-fold) — AUC-ROC:")
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
print(f"  Média : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# =============================================================================
# 6. VISUALIZAÇÕES
# =============================================================================
print("\n[6/6] Gerando visualizações...")

# --- Gráfico 11: Curva ROC ---
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(fpr, tpr, color='#C44E52', lw=2.5, label=f'Random Forest (AUC = {auc_roc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Classificador aleatório (AUC = 0.50)')
ax.fill_between(fpr, tpr, alpha=0.08, color='#C44E52')
ax.set_xlabel('Taxa de Falsos Positivos (1 − Especificidade)')
ax.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)')
ax.set_title('Curva ROC — Random Forest\nPredição de Doença Cardiovascular', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('graficos/11_curva_roc.png', bbox_inches='tight')
plt.close()

# --- Gráfico 12: Matriz de Confusão ---
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Sem doença', 'Com doença'], yticklabels=['Sem doença', 'Com doença'],
            annot_kws={'size': 14, 'weight': 'bold'})
ax.set_xlabel('Predito')
ax.set_ylabel('Real')
ax.set_title('Matriz de Confusão — Random Forest', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('graficos/12_matriz_confusao.png', bbox_inches='tight')
plt.close()

# --- Gráfico 13: Importância das Features ---
feature_names = continuous_cols + passthrough_cols
importances = pipeline.named_steps['model'].feature_importances_
feat_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(feat_df['feature'], feat_df['importance'], color=['#4C72B0' if i < len(feat_df)-3 else '#C44E52' for i in range(len(feat_df))])
ax.set_title('Importância das Features — Random Forest', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('graficos/13_feature_importance.png', bbox_inches='tight')
plt.close()
print("  ✓ Gráficos salvos com sucesso na pasta 'graficos/'.")


# =============================================================================
# EXPORTAÇÃO PARA O FORMATO ONNX (PARA O SEU SERVIDOR JAVA)
# =============================================================================
print("\n" + "=" * 65)
print("[BÔNUS] Exportando o modelo para formato ONNX...")
print("=" * 65)

# 1. Pegamos apenas os dados de treino já transformados pelo preprocessor do pipeline
X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)

# 2. Definimos o formato que a matriz ONNX espera receber (1 linha por 12 colunas)
dados_exemplo = X_train_transformed[:1].astype(np.float32)

# 3. Fazemos a conversão do modelo Random Forest final
modelo_random_forest = pipeline.named_steps['model']
modelo_onnx = to_onnx(modelo_random_forest, X=dados_exemplo)

# 4. Salvamos o arquivo mágico final
caminho_onnx = "modelo_cardio.onnx"
with open(caminho_onnx, "wb") as f:
    f.write(modelo_onnx.SerializeToString())

print(f"  ✓ Sucesso! Arquivo '{caminho_onnx}' gerado na raiz da pasta.")
print("=" * 65)