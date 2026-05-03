# =============================================================================
# ETAPA 3 — PRÉ-PROCESSAMENTO, MODELAGEM E AVALIAÇÃO
# Projeto: Predição de Doença Cardiovascular
# Dataset: Cardiovascular Disease Dataset (Cardio Train) — Kaggle
# Algoritmo: Random Forest
# Métrica principal: AUC-ROC
# =============================================================================
# INSTRUÇÕES:
#   1. Certifique-se de ter o arquivo 'cardio_train.csv' na raiz do projeto
#   2. Execute a partir da raiz: python src/pipeline_etapa3.py
#   3. Os gráficos serão salvos em 'graficos/'
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
# 'id' é apenas identificador do paciente — não carrega informação preditiva.
# Mantê-lo causaria data leakage (o modelo poderia "decorar" IDs).
df = df.drop(columns=['id'])
print("  ✓ Coluna 'id' removida (não é feature preditiva).")

# --- 2.2 Conversão de idade: dias → anos ---
# O dataset armazena idade em dias. Converter para anos torna o valor
# interpretável e comparável com faixas etárias clínicas.
df['age'] = (df['age'] / 365.25).round(1)
df.rename(columns={'age': 'age_years'}, inplace=True)
print("  ✓ Idade convertida de dias para anos (age_years = age / 365.25).")

# --- 2.3 Engenharia de feature: IMC ---
# O Índice de Massa Corporal combina peso e altura em um único índice clínico
# amplamente utilizado para avaliar risco cardiovascular.
# Fórmula: IMC = peso(kg) / altura(m)²
df['IMC'] = (df['weight'] / ((df['height'] / 100) ** 2)).round(2)
print("  ✓ IMC calculado (IMC = weight / (height/100)²).")

# --- 2.4 Remoção de registros clinicamente inválidos ---
# Registros com valores biologicamente impossíveis são erros de entrada,
# não outliers legítimos — portanto removemos ao invés de imputar.
# Critérios baseados em limites fisiológicos humanos.
mask_invalidos = (
    (df['ap_hi'] <= 0)    | (df['ap_hi'] > 300)   |   # pressão sistólica impossível
    (df['ap_lo'] <= 0)    | (df['ap_lo'] > 200)   |   # pressão diastólica impossível
    (df['ap_hi'] < df['ap_lo'])                    |   # sistólica < diastólica: impossível
    (df['height'] < 100)  | (df['height'] > 220)  |   # altura fora do range humano
    (df['weight'] < 30)   | (df['weight'] > 200)       # peso fora do range humano
)

n_invalidos = mask_invalidos.sum()
df_clean = df[~mask_invalidos].copy()

print(f"\n  LIMPEZA DE REGISTROS INVÁLIDOS:")
print(f"  ├── Registros originais     : {n_original:,}")
print(f"  ├── Registros removidos     : {n_invalidos:,} ({n_invalidos/n_original*100:.2f}%)")
print(f"  └── Registros após limpeza  : {len(df_clean):,}")

print("\n  Detalhamento por critério de remoção:")
criterios = {
    'ap_hi <= 0 ou > 300 mmHg'      : (df['ap_hi'] <= 0) | (df['ap_hi'] > 300),
    'ap_lo <= 0 ou > 200 mmHg'      : (df['ap_lo'] <= 0) | (df['ap_lo'] > 200),
    'ap_hi < ap_lo (fisicamente imp.)': df['ap_hi'] < df['ap_lo'],
    'height < 100 ou > 220 cm'      : (df['height'] < 100) | (df['height'] > 220),
    'weight < 30 ou > 200 kg'       : (df['weight'] < 30) | (df['weight'] > 200),
}
for criterio, mask in criterios.items():
    print(f"    {criterio:42s}: {mask.sum():,} registros")

# --- 2.5 Verificação de valores nulos ---
nulos = df_clean.isnull().sum().sum()
print(f"\n  Valores nulos após limpeza: {nulos} (dataset completo — nenhuma imputação necessária)")

# --- 2.6 Verificação do balanceamento após limpeza ---
print(f"\n  Balanceamento da variável alvo após limpeza:")
vc = df_clean['cardio'].value_counts()
for val, cnt in vc.items():
    label = 'Sem doença' if val == 0 else 'Com doença'
    print(f"    {label}: {cnt:,} ({cnt/len(df_clean)*100:.1f}%)")


# =============================================================================
# 3. SEPARAÇÃO FEATURES / ALVO E DIVISÃO TREINO/TESTE
# =============================================================================
print("\n[3/6] Separando features e dividindo os dados...")

X = df_clean.drop(columns=['cardio'])
y = df_clean['cardio']

# Variáveis contínuas: serão normalizadas com StandardScaler.
# Justificativa: embora o Random Forest não seja sensível à escala
# (ele usa thresholds de divisão, não distâncias), incluir a normalização
# torna o pipeline reutilizável para outros algoritmos (ex: Regressão Logística,
# SVM) sem modificações. É uma boa prática de engenharia de pipelines.
continuous_cols = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'IMC']

# Variáveis categóricas/binárias: já estão codificadas numericamente.
# cholesterol e gluc são ordinais (1 < 2 < 3), o que é semântico e correto.
# gender (1=F, 2=M), smoke, alco, active são binárias — mantidas como estão.
passthrough_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

# Divisão estratificada 80/20.
# Justificativa do 80/20: com ~68k registros, 20% (~13.6k) é suficiente
# para uma estimativa confiável do desempenho. Estratificado preserva a
# proporção da classe alvo em ambos os conjuntos.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  ├── Treino : {len(X_train):,} registros ({len(X_train)/len(X)*100:.0f}%)")
print(f"  └── Teste  : {len(X_test):,} registros ({len(X_test)/len(X)*100:.0f}%)")

print(f"\n  Balanceamento preservado pela estratificação:")
for split_name, y_split in [('Treino', y_train), ('Teste ', y_test)]:
    sem = (y_split == 0).sum()
    com = (y_split == 1).sum()
    print(f"    {split_name} — sem doença: {sem:,} ({sem/len(y_split)*100:.1f}%)  |  "
          f"com doença: {com:,} ({com/len(y_split)*100:.1f}%)")


# =============================================================================
# 4. CONSTRUÇÃO DO PIPELINE E TREINAMENTO
# =============================================================================
print("\n[4/6] Construindo e treinando o pipeline...")
print("-" * 40)

# ColumnTransformer: aplica transformações diferentes por grupo de colunas.
# 'passthrough' mantém as colunas categóricas sem transformação.
preprocessor = ColumnTransformer(transformers=[
    ('scaler',      StandardScaler(), continuous_cols),
    ('passthrough', 'passthrough',    passthrough_cols)
], remainder='drop')

# Random Forest — justificativa da escolha:
# 1. Desempenho: em todos os trabalhos correlatos analisados, Random Forest
#    obteve o melhor resultado para predição de doenças cardiovasculares (~87-90%).
# 2. Robustez: lida naturalmente com variáveis de escalas e tipos diferentes
#    (contínuas, ordinais, binárias) sem necessidade de tratamentos adicionais.
# 3. Interpretabilidade parcial: fornece importância das features, permitindo
#    entender quais variáveis mais contribuem para a predição.
# 4. Resistência a overfitting: o ensemble de árvores reduz a variância.
#
# Hiperparâmetros:
# - n_estimators=100: número de árvores. 100 é o padrão consolidado na literatura
#   — oferece boa estabilidade sem custo computacional excessivo.
# - random_state=42: garante reprodutibilidade dos resultados.
# - n_jobs=-1: usa todos os núcleos disponíveis para treino paralelo.
# - max_features='sqrt': padrão do sklearn para classificação — em cada divisão,
#   considera raiz quadrada do total de features (evita correlação entre árvores).
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
y_proba = pipeline.predict_proba(X_test)[:, 1]  # probabilidade da classe positiva

# Cálculo das métricas
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

  ★  AUC-ROC é a métrica principal porque mede a capacidade discriminativa
     do modelo em todos os limiares de decisão simultaneamente. Interpretação:
     o modelo tem {auc_roc*100:.1f}% de probabilidade de classificar corretamente
     um paciente doente acima de um saudável em uma comparação aleatória.
     Adequada para datasets balanceados e independente do limiar de corte.

  ★★ Recall destacado secundariamente pela relevância clínica: um falso
     negativo (doença não detectada) é clinicamente mais grave que um
     falso positivo (alarme desnecessário).
""")

print("  RELATÓRIO COMPLETO POR CLASSE:")
print(classification_report(y_test, y_pred, target_names=['Sem doença (0)', 'Com doença (1)']))

# Matriz de confusão — valores absolutos para discussão
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"  MATRIZ DE CONFUSÃO:")
print(f"  ├── Verdadeiros Negativos (TN): {tn:,}  — sem doença, predito corretamente")
print(f"  ├── Falsos Positivos     (FP): {fp:,}  — sem doença, predito como doente")
print(f"  ├── Falsos Negativos     (FN): {fn:,}  — doente, predito como saudável ⚠")
print(f"  └── Verdadeiros Positivos (TP): {tp:,}  — doente, predito corretamente")

# Validação cruzada estratificada (5-fold) — AUC-ROC
# Justificativa: a divisão simples treino/teste pode ter variância.
# A validação cruzada estima o desempenho em 5 partições diferentes,
# fornecendo média e desvio padrão — indica se o modelo é estável.
print(f"\n  VALIDAÇÃO CRUZADA ESTRATIFICADA (5-fold) — AUC-ROC:")
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
for i, s in enumerate(cv_scores, 1):
    print(f"    Fold {i}: {s:.4f}")
print(f"  Média : {cv_scores.mean():.4f}")
print(f"  Desvio: {cv_scores.std():.4f}")
if cv_scores.std() < 0.01:
    print("  → Baixo desvio padrão: modelo estável entre as partições ✓")


# =============================================================================
# 6. VISUALIZAÇÕES
# =============================================================================
print("\n[6/6] Gerando visualizações...")

# --- Gráfico 11: Curva ROC ---
# Mostra o trade-off entre sensibilidade e especificidade em todos os limiares.
# A área sob a curva (AUC) resume o desempenho em um único número.
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(fpr, tpr, color='#C44E52', lw=2.5,
        label=f'Random Forest (AUC = {auc_roc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1.5,
        label='Classificador aleatório (AUC = 0.50)')
ax.fill_between(fpr, tpr, alpha=0.08, color='#C44E52')
ax.set_xlabel('Taxa de Falsos Positivos (1 − Especificidade)', fontsize=12)
ax.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=12)
ax.set_title('Curva ROC — Random Forest\nPredição de Doença Cardiovascular',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.02])
plt.tight_layout()
plt.savefig('graficos/11_curva_roc.png', bbox_inches='tight')
plt.close()
print("  ✓ graficos/11_curva_roc.png")

# --- Gráfico 12: Matriz de Confusão ---
# Visualiza TN, FP, FN, TP para identificar onde o modelo erra.
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Sem doença', 'Com doença'],
            yticklabels=['Sem doença', 'Com doença'],
            annot_kws={'size': 14, 'weight': 'bold'})
ax.set_xlabel('Predito', fontsize=12)
ax.set_ylabel('Real', fontsize=12)
ax.set_title('Matriz de Confusão — Random Forest', fontsize=13, fontweight='bold')
ax.text(0.5, -0.12,
        f'TN={tn:,}  |  FP={fp:,}  |  FN={fn:,}  |  TP={tp:,}',
        transform=ax.transAxes, ha='center', fontsize=10, color='gray')
plt.tight_layout()
plt.savefig('graficos/12_matriz_confusao.png', bbox_inches='tight')
plt.close()
print("  ✓ graficos/12_matriz_confusao.png")

# --- Gráfico 13: Importância das Features ---
# Mede a redução média de impureza (Gini) causada por cada feature.
# Features com maior importância contribuem mais para as decisões do modelo.
feature_names = continuous_cols + passthrough_cols
importances = pipeline.named_steps['model'].feature_importances_
feat_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feat_df = feat_df.sort_values('importance', ascending=True).reset_index(drop=True)

# Top 3 features em vermelho, demais em azul
n = len(feat_df)
colors = ['#C44E52' if i >= n - 3 else '#4C72B0' for i in range(n)]

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(feat_df['feature'], feat_df['importance'], color=colors, edgecolor='white')
ax.set_xlabel('Importância (Mean Decrease in Gini Impurity)', fontsize=11)
ax.set_title('Importância das Features — Random Forest', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
for i, val in enumerate(feat_df['importance']):
    ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9)
ax.legend(handles=[
    plt.Rectangle((0,0),1,1, color='#C44E52', label='Top 3 features'),
    plt.Rectangle((0,0),1,1, color='#4C72B0', label='Demais features')
], fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig('graficos/13_feature_importance.png', bbox_inches='tight')
plt.close()
print("  ✓ graficos/13_feature_importance.png")


# =============================================================================
# RESUMO FINAL DO PIPELINE
# =============================================================================
print("\n" + "=" * 65)
print("  RESUMO DA ETAPA 3 — PIPELINE COMPLETO")
print("=" * 65)
print(f"""
  PROBLEMA
  └── Classificação binária: predizer presença de doença cardiovascular

  PRÉ-PROCESSAMENTO
  ├── Remoção do ID                  : coluna não preditiva
  ├── Conversão de idade             : dias → anos (age / 365.25)
  ├── Feature engineering            : IMC = peso / (altura/100)²
  ├── Limpeza de inválidos           : {n_invalidos:,} registros removidos ({n_invalidos/n_original*100:.2f}%)
  ├── Normalização (contínuas)       : StandardScaler
  └── Codificação (categóricas)      : já numéricas — sem alteração

  DIVISÃO DOS DADOS
  ├── Treino : {len(X_train):,} registros (80%) — estratificado
  └── Teste  : {len(X_test):,} registros (20%) — estratificado

  MODELO: Random Forest
  ├── n_estimators                   : 100 árvores
  ├── max_features                   : sqrt (padrão sklearn)
  └── random_state                   : 42 (reprodutibilidade)

  AVALIAÇÃO — CONJUNTO DE TESTE
  ├── AUC-ROC  ★ (principal)        : {auc_roc:.4f}
  ├── Acurácia                       : {accuracy:.4f}
  ├── Precisão                       : {precision:.4f}
  ├── Recall ★★ (clínico)           : {recall:.4f}
  └── F1-Score                       : {f1:.4f}

  VALIDAÇÃO CRUZADA (5-fold AUC-ROC)
  └── {cv_scores.mean():.4f} ± {cv_scores.std():.4f}

  GRÁFICOS GERADOS:
  11 — Curva ROC
  12 — Matriz de Confusão
  13 — Importância das Features
""")
print("=" * 65)
print("  Etapa 3 concluída com sucesso!")
print("=" * 65)
