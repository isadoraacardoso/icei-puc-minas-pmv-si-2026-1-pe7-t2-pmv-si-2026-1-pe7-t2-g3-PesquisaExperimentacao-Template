# =============================================================================
# ETAPA 3 — PRÉ-PROCESSAMENTO, MODELAGEM E AVALIAÇÃO
# Projeto: Predição de Doença Cardiovascular
# Dataset: Cardiovascular Disease Dataset (Cardio Train) — Kaggle
# Algoritmo: Random Forest
# Métrica principal: AUC-ROC
# =============================================================================
# INSTRUÇÕES:
#   1. Certifique-se de ter o arquivo 'cardio_train_sem_valores_invalidos.csv'
#      na mesma pasta deste script.
#   2. Execute: python pipeline_etapa3.py
#   3. Os gráficos e métricas serão salvos em 'graficos/', separados por divisão:
#      - graficos/80_20/
#      - graficos/75_25/
#      - graficos/70_30/
# =============================================================================

from pathlib import Path
import os
import warnings

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_theme(style="whitegrid", palette="muted")

# =============================================================================
# 0. CONFIGURAÇÃO DE CAMINHOS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = BASE_DIR / "cardio_train_sem_valores_invalidos.csv"
GRAFICOS_DIR = BASE_DIR / "graficos"

GRAFICOS_DIR.mkdir(exist_ok=True)

# Pastas específicas para cada divisão treino/teste
PASTAS_TESTES = {
    "80_20": GRAFICOS_DIR / "80_20",
    "75_25": GRAFICOS_DIR / "75_25",
    "70_30": GRAFICOS_DIR / "70_30",
}

for pasta in PASTAS_TESTES.values():
    pasta.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("  ETAPA 3 — PRÉ-PROCESSAMENTO E MODELAGEM")
print("  Predição de Doença Cardiovascular — Random Forest")
print("=" * 65)


# =============================================================================
# 1. CARREGAMENTO DOS DADOS
# =============================================================================
print("\n[1/6] Carregando o dataset...")

if not ARQUIVO_ENTRADA.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {ARQUIVO_ENTRADA}\n"
        "Verifique se 'cardio_train_sem_valores_invalidos.csv' está na mesma pasta deste script."
    )

df = pd.read_csv(ARQUIVO_ENTRADA, sep=';')
print(f"  ✓ Dataset carregado: {df.shape[0]:,} registros × {df.shape[1]} colunas")

n_original = len(df)


# =============================================================================
# 2. PRÉ-PROCESSAMENTO
# =============================================================================
print("\n[2/6] Pré-processamento dos dados...")
print("-" * 40)

# --- 2.1 Remoção do ID ---
# 'id' é apenas identificador do paciente — não carrega informação preditiva.
# Mantê-lo poderia fazer o modelo aprender padrões artificiais.
df = df.drop(columns=['id'], errors='ignore')
print("  ✓ Coluna 'id' removida quando existente (não é feature preditiva).")

# --- 2.2 Conversão de idade: dias → anos ---
# O dataset original armazena idade em dias. Converter para anos torna o valor
# interpretável e comparável com faixas etárias clínicas.
# Caso a base já possua 'age_years', ela será atualizada a partir de 'age' quando possível.
if 'age' in df.columns:
    df['age_years'] = (df['age'] / 365.25).round(1)
    df = df.drop(columns=['age'])
    print("  ✓ Idade convertida de dias para anos (age_years = age / 365.25).")
elif 'age_years' in df.columns:
    print("  ✓ Coluna 'age_years' já existente na base; conversão de idade não foi necessária.")
else:
    raise ValueError("A base precisa conter a coluna 'age' ou 'age_years'.")

# --- 2.3 Engenharia de feature: IMC ---
# O Índice de Massa Corporal combina peso e altura em um único índice clínico
# amplamente utilizado para avaliar risco cardiovascular.
# Fórmula: IMC = peso(kg) / altura(m)².
df['IMC'] = (df['weight'] / ((df['height'] / 100) ** 2)).round(2)
df = df.drop(columns=['bmi'], errors='ignore')
print("  ✓ IMC calculado (IMC = weight / (height/100)²).")

# --- 2.4 Validação de registros clinicamente inválidos ---
# A base lida já deve estar com limpeza dos valores clinicamente inválidos.
# Esta checagem é mantida como segurança para garantir reprodutibilidade do pipeline.
# Registros com valores biologicamente impossíveis são erros de entrada.
mask_invalidos = (
    (df['ap_hi'] <= 0)    | (df['ap_hi'] > 300)   |   # pressão sistólica impossível
    (df['ap_lo'] <= 0)    | (df['ap_lo'] > 200)   |   # pressão diastólica impossível
    (df['ap_hi'] < df['ap_lo'])                    |   # sistólica < diastólica: impossível
    (df['height'] < 100)  | (df['height'] > 220)  |   # altura fora do range humano
    (df['weight'] < 30)   | (df['weight'] > 200)       # peso fora do range humano
)

n_invalidos = int(mask_invalidos.sum())
df_clean = df[~mask_invalidos].copy()

print(f"\n  LIMPEZA DOS VALORES CLINICAMENTE INVÁLIDOS:")
print(f"  ├── Registros originais     : {n_original:,}")
print(f"  ├── Registros removidos     : {n_invalidos:,} ({n_invalidos/n_original*100:.2f}%)")
print(f"  └── Registros após limpeza  : {len(df_clean):,}")

print("\n  Detalhamento por critério de remoção:")
criterios = {
    'ap_hi <= 0 ou > 300 mmHg'        : (df['ap_hi'] <= 0) | (df['ap_hi'] > 300),
    'ap_lo <= 0 ou > 200 mmHg'        : (df['ap_lo'] <= 0) | (df['ap_lo'] > 200),
    'ap_hi < ap_lo (fisicamente imp.)': df['ap_hi'] < df['ap_lo'],
    'height < 100 ou > 220 cm'        : (df['height'] < 100) | (df['height'] > 220),
    'weight < 30 ou > 200 kg'         : (df['weight'] < 30) | (df['weight'] > 200),
}
for criterio, mask in criterios.items():
    print(f"    {criterio:42s}: {int(mask.sum()):,} registros")

# --- 2.5 Verificação de valores nulos ---
nulos = int(df_clean.isnull().sum().sum())
print(f"\n  Valores nulos após limpeza: {nulos} (dataset completo — nenhuma imputação necessária)")

# --- 2.6 Verificação do balanceamento após limpeza ---
print(f"\n  Balanceamento da variável alvo após limpeza:")
vc = df_clean['cardio'].value_counts()
for val, cnt in vc.items():
    label = 'Sem doença' if val == 0 else 'Com doença'
    print(f"    {label}: {cnt:,} ({cnt/len(df_clean)*100:.1f}%)")


# =============================================================================
# 3. SEPARAÇÃO FEATURES / ALVO
# =============================================================================
print("\n[3/6] Separando features e alvo...")

X = df_clean.drop(columns=['cardio'])
y = df_clean['cardio']

# Variáveis contínuas: serão normalizadas com StandardScaler.
# Embora Random Forest não dependa de escala, a normalização torna o pipeline
# reaproveitável para algoritmos sensíveis à escala.
continuous_cols = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'IMC']

# Variáveis categóricas/binárias: já estão codificadas numericamente.
passthrough_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']


# =============================================================================
# 4. FUNÇÕES DO PIPELINE, TREINAMENTO E AVALIAÇÃO
# =============================================================================
print("\n[4/6] Configurando pipeline e cenários de treino/teste...")
print("-" * 40)


def criar_pipeline():
    """
    Cria um pipeline novo para cada experimento.
    Isso evita reaproveitar um modelo já treinado entre diferentes divisões treino/teste.
    """
    preprocessor = ColumnTransformer(transformers=[
        ('scaler',      StandardScaler(), continuous_cols),
        ('passthrough', 'passthrough',    passthrough_cols)
    ], remainder='drop')

    model = RandomForestClassifier(
        n_estimators=100,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model',        model)
    ])


def avaliar_divisao(nome_divisao, test_size, pasta_saida):
    """
    Executa a divisão treino/teste, treina o Random Forest,
    calcula métricas, salva CSVs e gera gráficos para uma divisão específica.
    """
    train_pct = int(round((1 - test_size) * 100))
    test_pct = int(round(test_size * 100))

    print(f"\n[Experimento {nome_divisao}] Divisão treino/teste {train_pct}/{test_pct}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"  ├── Treino : {len(X_train):,} registros ({len(X_train)/len(X)*100:.0f}%)")
    print(f"  └── Teste  : {len(X_test):,} registros ({len(X_test)/len(X)*100:.0f}%)")

    print("  Balanceamento preservado pela estratificação:")
    for split_name, y_split in [('Treino', y_train), ('Teste ', y_test)]:
        sem = int((y_split == 0).sum())
        com = int((y_split == 1).sum())
        print(f"    {split_name} — sem doença: {sem:,} ({sem/len(y_split)*100:.1f}%)  |  "
              f"com doença: {com:,} ({com/len(y_split)*100:.1f}%)")

    pipeline = criar_pipeline()

    print("  Treinando Random Forest (100 árvores)...")
    pipeline.fit(X_train, y_train)
    print("  ✓ Pipeline treinado com sucesso.")

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_proba)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"""
  MÉTRICAS DE DESEMPENHO — divisão {nome_divisao}:

  ┌─────────────────────────────────────────────┐
  │  AUC-ROC  ★ (métrica principal)  :  {auc_roc:.4f}  │
  │  Acurácia                        :  {accuracy:.4f}  │
  │  Precisão                        :  {precision:.4f}  │
  │  Recall (Sensibilidade) ★★       :  {recall:.4f}  │
  │  F1-Score                        :  {f1:.4f}  │
  └─────────────────────────────────────────────┘
""")

    print("  RELATÓRIO COMPLETO POR CLASSE:")
    print(classification_report(y_test, y_pred, target_names=['Sem doença (0)', 'Com doença (1)']))

    print("  MATRIZ DE CONFUSÃO:")
    print(f"  ├── Verdadeiros Negativos (TN): {tn:,}  — sem doença, predito corretamente")
    print(f"  ├── Falsos Positivos     (FP): {fp:,}  — sem doença, predito como doente")
    print(f"  ├── Falsos Negativos     (FN): {fn:,}  — doente, predito como saudável ⚠")
    print(f"  └── Verdadeiros Positivos (TP): {tp:,}  — doente, predito corretamente")

    print("\n  VALIDAÇÃO CRUZADA ESTRATIFICADA (5-fold) — AUC-ROC:")
    cv_pipeline = criar_pipeline()
    cv_scores = cross_val_score(cv_pipeline, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
    for i, s in enumerate(cv_scores, 1):
        print(f"    Fold {i}: {s:.4f}")
    print(f"  Média : {cv_scores.mean():.4f}")
    print(f"  Desvio: {cv_scores.std():.4f}")

    # -------------------------------------------------------------------------
    # Salvamento das métricas em CSV
    # -------------------------------------------------------------------------
    metricas = pd.DataFrame([{
        'divisao': nome_divisao,
        'treino_percentual': train_pct,
        'teste_percentual': test_pct,
        'registros_total': len(X),
        'registros_treino': len(X_train),
        'registros_teste': len(X_test),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std()
    }])

    metricas.to_csv(pasta_saida / f"metricas_random_forest_{nome_divisao}.csv", sep=';', index=False)

    relatorio_dict = classification_report(
        y_test,
        y_pred,
        target_names=['Sem doença (0)', 'Com doença (1)'],
        output_dict=True
    )
    df_relatorio = pd.DataFrame(relatorio_dict).T
    df_relatorio.to_csv(pasta_saida / f"classification_report_{nome_divisao}.csv", sep=';', index=True)

    df_cm = pd.DataFrame(
        cm,
        index=['Real: Sem doença', 'Real: Com doença'],
        columns=['Predito: Sem doença', 'Predito: Com doença']
    )
    df_cm.to_csv(pasta_saida / f"matriz_confusao_{nome_divisao}.csv", sep=';', index=True)

    # -------------------------------------------------------------------------
    # Gráfico 1: Curva ROC
    # -------------------------------------------------------------------------
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color='#C44E52', lw=2.5,
            label=f'Random Forest (AUC = {auc_roc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5,
            label='Classificador aleatório (AUC = 0.50)')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#C44E52')
    ax.set_xlabel('Taxa de Falsos Positivos (1 − Especificidade)', fontsize=12)
    ax.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=12)
    ax.set_title(f'Curva ROC — Random Forest\nDivisão {nome_divisao}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    plt.tight_layout()
    plt.savefig(pasta_saida / f"curva_roc_{nome_divisao}.png", bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------------------
    # Gráfico 2: Matriz de Confusão
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Sem doença', 'Com doença'],
                yticklabels=['Sem doença', 'Com doença'],
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_xlabel('Predito', fontsize=12)
    ax.set_ylabel('Real', fontsize=12)
    ax.set_title(f'Matriz de Confusão — Random Forest\nDivisão {nome_divisao}',
                 fontsize=13, fontweight='bold')
    ax.text(0.5, -0.12,
            f'TN={tn:,}  |  FP={fp:,}  |  FN={fn:,}  |  TP={tp:,}',
            transform=ax.transAxes, ha='center', fontsize=10, color='gray')
    plt.tight_layout()
    plt.savefig(pasta_saida / f"matriz_confusao_{nome_divisao}.png", bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------------------
    # Gráfico 3: Importância das Features
    # -------------------------------------------------------------------------
    feature_names = continuous_cols + passthrough_cols
    importances = pipeline.named_steps['model'].feature_importances_
    feat_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feat_df = feat_df.sort_values('importance', ascending=True).reset_index(drop=True)

    n = len(feat_df)
    colors = ['#C44E52' if i >= n - 3 else '#4C72B0' for i in range(n)]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(feat_df['feature'], feat_df['importance'], color=colors, edgecolor='white')
    ax.set_xlabel('Importância (Mean Decrease in Gini Impurity)', fontsize=11)
    ax.set_title(f'Importância das Features — Random Forest\nDivisão {nome_divisao}',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for i, val in enumerate(feat_df['importance']):
        ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9)
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color='#C44E52', label='Top 3 features'),
        plt.Rectangle((0, 0), 1, 1, color='#4C72B0', label='Demais features')
    ], fontsize=9, loc='lower right')
    plt.tight_layout()
    plt.savefig(pasta_saida / f"feature_importance_{nome_divisao}.png", bbox_inches='tight')
    plt.close()

    feat_df.to_csv(pasta_saida / f"feature_importance_{nome_divisao}.csv", sep=';', index=False)

    print(f"  ✓ Arquivos da divisão {nome_divisao} salvos em: {pasta_saida}")

    return metricas


# =============================================================================
# 5. TREINAMENTO E AVALIAÇÃO EM DIFERENTES DIVISÕES
# =============================================================================
print("\n[5/6] Treinando e avaliando o modelo em 3 divisões treino/teste...")
print("-" * 40)

experimentos = [
    ("80_20", 0.20, PASTAS_TESTES["80_20"]),
    ("75_25", 0.25, PASTAS_TESTES["75_25"]),
    ("70_30", 0.30, PASTAS_TESTES["70_30"]),
]

metricas_todas = []
for nome_divisao, test_size, pasta_saida in experimentos:
    metricas_todas.append(avaliar_divisao(nome_divisao, test_size, pasta_saida))

# CSV consolidado com os resultados das três divisões.
df_metricas_todas = pd.concat(metricas_todas, ignore_index=True)
df_metricas_todas.to_csv(
    GRAFICOS_DIR / "metricas_random_forest_treino_teste.csv",
    sep=';',
    index=False
)

print(f"\n  ✓ Métricas consolidadas salvas em:")
print(f"    {GRAFICOS_DIR / 'metricas_random_forest_treino_teste.csv'}")


# =============================================================================
# 6. RESUMO FINAL DO PIPELINE
# =============================================================================
print("\n" + "=" * 65)
print("  RESUMO DA ETAPA 3 — PIPELINE COMPLETO")
print("=" * 65)
print(f"""
  PROBLEMA
  └── Classificação binária: predizer presença de doença cardiovascular

  PRÉ-PROCESSAMENTO
  ├── Remoção do ID                              : coluna não preditiva
  ├── Conversão de idade                         : dias → anos (age / 365.25)
  ├── Feature engineering                        : IMC = peso / (altura/100)²
  ├── Limpeza dos valores clinicamente inválidos : {n_invalidos:,} registros removidos ({n_invalidos/n_original*100:.2f}%)
  ├── Normalização (contínuas)                   : StandardScaler
  └── Codificação (categóricas)                  : já numéricas — sem alteração

  DIVISÕES DOS DADOS AVALIADAS
  ├── 80/20 — treino/teste estratificado
  ├── 75/25 — treino/teste estratificado
  └── 70/30 — treino/teste estratificado

  MODELO: Random Forest
  ├── n_estimators                               : 100 árvores
  ├── max_features                               : sqrt
  └── random_state                               : 42 (reprodutibilidade)

  MÉTRICA PRINCIPAL
  └── AUC-ROC, pois avalia a capacidade discriminativa do modelo
      em diferentes limiares de decisão.

  ARQUIVOS GERADOS
  ├── graficos/metricas_random_forest_treino_teste.csv
  ├── graficos/80_20/
  ├── graficos/75_25/
  └── graficos/70_30/
""")

print("=" * 65)
print("  Etapa 3 concluída com sucesso!")
print("=" * 65)
