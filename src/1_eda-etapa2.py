# =============================================================================
# ETAPA 2 — ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
# Projeto: Predição de Doença Cardiovascular
# Dataset: Cardiovascular Disease Dataset (Cardio Train) — Kaggle
# =============================================================================
# INSTRUÇÕES:
#   1. Baixe o arquivo 'cardio_train.csv' em:
#      https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
#   2. Salve o CSV na mesma pasta deste script
#   3. Execute: python eda_cardiovascular.py
#   4. Os gráficos serão salvos automaticamente na pasta 'graficos/'
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings
from scipy import stats
from pathlib import Path

warnings.filterwarnings('ignore')

# Configurações visuais
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_theme(style="whitegrid", palette="muted")

# Criar pasta para salvar gráficos
os.makedirs('graficos', exist_ok=True)

print("=" * 65)
print("  ETAPA 2 — ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
print("  Predição de Doença Cardiovascular")
print("=" * 65)


# =============================================================================
# 1. CARREGAMENTO DOS DADOS
# =============================================================================
print("\n[1/8] Carregando o dataset...")

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / 'cardio_train.csv'

try:
    df = pd.read_csv(CSV_PATH, sep=';')
    print(f"  ✓ Dataset carregado com sucesso.")
except FileNotFoundError:
    print("  ✗ ERRO: Arquivo 'cardio_train.csv' não encontrado.")
    print("    Baixe em: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset")
    exit()


# =============================================================================
# 2. INSPEÇÃO ESTRUTURAL
# =============================================================================
print("\n[2/8] Inspeção estrutural do dataset")
print("-" * 40)

print(f"\n  Dimensões: {df.shape[0]:,} linhas × {df.shape[1]} colunas")

print("\n  PRIMEIRAS 5 LINHAS:")
print(df.head().to_string())

print("\n  ÚLTIMAS 5 LINHAS:")
print(df.tail().to_string())

print("\n  TIPOS DE DADOS E VALORES NULOS:")
info_df = pd.DataFrame({
    'Tipo': df.dtypes,
    'Nulos': df.isnull().sum(),
    'Nulos (%)': (df.isnull().sum() / len(df) * 100).round(2),
    'Únicos': df.nunique()
})
print(info_df.to_string())

print("\n  REGISTROS DUPLICADOS:", df.duplicated().sum())


# =============================================================================
# 3. PREPARAÇÃO DAS VARIÁVEIS (para análise — sem alterar o df original)
# =============================================================================
print("\n[3/8] Preparando variáveis para análise...")

# Criar cópia de trabalho com idade em anos (facilita interpretação dos gráficos)
df_plot = df.copy()
df_plot['age_years'] = (df_plot['age'] / 365.25).round(1)

# Mapeamentos para legibilidade nos gráficos
df_plot['gender_label'] = df_plot['gender'].map({1: 'Feminino', 2: 'Masculino'})
df_plot['cardio_label'] = df_plot['cardio'].map({0: 'Sem doença', 1: 'Com doença'})
df_plot['cholesterol_label'] = df_plot['cholesterol'].map({
    1: 'Normal', 2: 'Acima do normal', 3: 'Muito acima'})
df_plot['gluc_label'] = df_plot['gluc'].map({
    1: 'Normal', 2: 'Acima do normal', 3: 'Muito acima'})

# IMC calculado
df_plot['bmi'] = (df_plot['weight'] / ((df_plot['height'] / 100) ** 2)).round(2)

print("  ✓ Variáveis de apoio criadas (sem alterar o dataset original).")


# =============================================================================
# 4. ESTATÍSTICAS DESCRITIVAS
# =============================================================================
print("\n[4/8] Estatísticas descritivas")
print("-" * 40)

# Variáveis numéricas para análise descritiva (com idade em anos)
df_desc = df_plot[['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']].copy()
df_desc.columns = ['Idade (anos)', 'Altura (cm)', 'Peso (kg)',
                   'Pressão Sistólica', 'Pressão Diastólica', 'IMC']

desc = df_desc.describe().T
desc['mediana'] = df_desc.median()
desc['moda'] = df_desc.mode().iloc[0]
desc['IQR'] = df_desc.quantile(0.75) - df_desc.quantile(0.25)
desc['assimetria'] = df_desc.skew()
desc['curtose'] = df_desc.kurtosis()

cols_print = ['count', 'mean', 'mediana', 'moda', 'std', 'IQR',
              'min', '25%', '75%', 'max', 'assimetria', 'curtose']

desc_final = desc[cols_print].round(2)

desc_final = desc_final.rename(columns={
    'count': 'qtd',
    'mean': 'media',
    'std': 'desvio padrão',
    'min': 'min',
    '25%': '25%',
    '75%': '75%',
    'max': 'max',
    'IQR': 'IQR'
})

desc_final.index.name = 'Variável'

print("\n  TABELA DE ESTATÍSTICAS DESCRITIVAS:")
print(desc_final.to_string())

# Salvar tabela como CSV
desc_final.to_csv('graficos/tabela_estatisticas_descritivas.csv')
print("\n  ✓ Tabela salva em 'graficos/tabela_estatisticas_descritivas.csv'")

# Variáveis categóricas
# Qual é a proporção de homens e mulheres?
# Quantas pessoas têm colesterol ou glicose normal ou elevado?
# Quantas pessoas fumam?
# Quantas consomem álcool?
# Quantas praticam atividade física?
# A variável alvo está balanceada?
print("\n  DISTRIBUIÇÃO DAS VARIÁVEIS CATEGÓRICAS:")
cat_vars = {
    'Gênero': df_plot['gender_label'],
    'Colesterol': df_plot['cholesterol_label'],
    'Glicose': df_plot['gluc_label'],
    'Tabagismo (1=sim)': df_plot['smoke'],
    'Álcool (1=sim)': df_plot['alco'],
    'Atividade física (1=sim)': df_plot['active'],
    'Doença cardiovascular (alvo)': df_plot['cardio_label']
}
for nome, serie in cat_vars.items():
    vc = serie.value_counts()
    pct = (vc / len(df) * 100).round(1)
    print(f"\n  {nome}:")
    for val in vc.index:
        print(f"    {val}: {vc[val]:,} ({pct[val]}%)")


# =============================================================================
# 5. VISUALIZAÇÕES — DISTRIBUIÇÕES
# =============================================================================
print("\n[5/8] Gerando gráficos de distribuição...")

# --- Gráfico 1: Histogramas + KDE das variáveis numéricas ---
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Distribuição das Variáveis Numéricas', fontsize=15, fontweight='bold', y=1.01)

variaveis = [
    ('age_years', 'Idade (anos)', '#4C72B0'),
    ('height', 'Altura (cm)', '#55A868'),
    ('weight', 'Peso (kg)', '#C44E52'),
    ('ap_hi', 'Pressão Sistólica (mmHg)', '#8172B2'),
    ('ap_lo', 'Pressão Diastólica (mmHg)', '#CCB974'),
    ('bmi', 'IMC (kg/m²)', '#64B5CD'),
]

for ax, (col, titulo, cor) in zip(axes.flatten(), variaveis):
    dados = df_plot[col].dropna()
    # Limitar outliers extremos apenas para visualização
    q1, q3 = dados.quantile(0.01), dados.quantile(0.99)
    dados_viz = dados[(dados >= q1) & (dados <= q3)]
    ax.hist(dados_viz, bins=40, color=cor, alpha=0.7, edgecolor='white', density=True)
    dados_viz.plot.kde(ax=ax, color='black', linewidth=1.5)
    ax.axvline(dados.mean(), color='red', linestyle='--', linewidth=1.2, label=f'Média: {dados.mean():.1f}')
    ax.axvline(dados.median(), color='blue', linestyle=':', linewidth=1.2, label=f'Mediana: {dados.median():.1f}')
    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel('')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graficos/01_histogramas_distribuicao.png', bbox_inches='tight')
plt.close()
print("  ✓ graficos/01_histogramas_distribuicao.png")

# --- Gráfico 2: Box plots das variáveis numéricas por presença de doença ---
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Box Plots por Presença de Doença Cardiovascular', fontsize=15, fontweight='bold')

palette = {'Sem doença': '#4C72B0', 'Com doença': '#C44E52'}

for ax, (col, titulo, _) in zip(axes.flatten(), variaveis):
    dados_bp = df_plot[[col, 'cardio_label']].copy().dropna()
    # Limitar outliers extremos apenas para visualização
    q1, q3 = dados_bp[col].quantile(0.01), dados_bp[col].quantile(0.99)
    dados_bp = dados_bp[(dados_bp[col] >= q1) & (dados_bp[col] <= q3)]
    sns.boxplot(data=dados_bp, x='cardio_label', y=col, palette=palette, ax=ax,
                order=['Sem doença', 'Com doença'])
    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('graficos/02_boxplots_por_cardio.png', bbox_inches='tight')
plt.close()
print("  ✓ graficos/02_boxplots_por_cardio.png")

# --- Gráfico 3: Distribuição da variável alvo ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Distribuição da Variável Alvo (Doença Cardiovascular)', fontsize=13, fontweight='bold')

contagens = df_plot['cardio_label'].value_counts()
cores = ['#4C72B0', '#C44E52']

axes[0].bar(contagens.index, contagens.values, color=cores, edgecolor='white', width=0.5)
for i, (val, cnt) in enumerate(zip(contagens.index, contagens.values)):
    axes[0].text(i, cnt + 300, f'{cnt:,}\n({cnt/len(df)*100:.1f}%)',
                 ha='center', fontsize=11, fontweight='bold')
axes[0].set_title('Contagem absoluta')
axes[0].set_ylabel('Número de pacientes')
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].pie(contagens.values, labels=contagens.index, colors=cores,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
axes[1].set_title('Proporção')

plt.tight_layout()
plt.savefig('graficos/03_distribuicao_variavel_alvo.png', bbox_inches='tight')
plt.close()
print("  ✓ graficos/03_distribuicao_variavel_alvo.png")

# --- Gráfico 4: Variáveis categóricas ---
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Distribuição das Variáveis Categóricas por Doença Cardiovascular',
             fontsize=14, fontweight='bold')

cat_plot = [
    ('gender_label', 'Gênero', None),
    ('cholesterol_label', 'Colesterol', ['Normal', 'Acima do normal', 'Muito acima']),
    ('gluc_label', 'Glicose', ['Normal', 'Acima do normal', 'Muito acima']),
    ('smoke', 'Tabagismo', None),
    ('alco', 'Consumo de Álcool', None),
    ('active', 'Atividade Física', None),
]

for ax, (col, titulo, ordem) in zip(axes.flatten(), cat_plot):
    if ordem:
        ct = pd.crosstab(df_plot[col], df_plot['cardio_label'])
        ct = ct.reindex(ordem)
    else:
        ct = pd.crosstab(df_plot[col], df_plot['cardio_label'])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.plot(kind='bar', ax=ax, color=['#4C72B0', '#C44E52'],
                edgecolor='white', rot=20)
    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel('')
    ax.set_ylabel('% dentro da categoria')
    ax.legend(title='', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('graficos/04_categoricas_por_cardio.png', bbox_inches='tight')
plt.close()
print("  ✓ graficos/04_categoricas_por_cardio.png")


# =============================================================================
# 6. ANÁLISE DE CORRELAÇÃO
# =============================================================================
print("\n[6/8] Análise de correlação...")

# Variáveis numéricas para correlação (incluindo IMC e idade em anos)
num_cols = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo',
            'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi', 'cardio']
labels_corr = ['Idade', 'Altura', 'Peso', 'P. Sistólica', 'P. Diastólica',
               'Colesterol', 'Glicose', 'Tabagismo', 'Álcool',
               'Ativ. Física', 'IMC', 'Doença Card.']

corr_matrix = df_plot[num_cols].corr(method='pearson')
corr_matrix.index = labels_corr
corr_matrix.columns = labels_corr

# Correlações com a variável alvo
corr_com_alvo = corr_matrix['Doença Card.'].drop('Doença Card.').sort_values(ascending=False)
print("\n  CORRELAÇÃO DE PEARSON COM A VARIÁVEL ALVO (cardio):")
for var, val in corr_com_alvo.items():
    barra = '█' * int(abs(val) * 30)
    sinal = '+' if val > 0 else '-'
    print(f"    {var:20s}: {sinal}{abs(val):.4f}  {barra}")

# Heatmap de correlação
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax,
            annot_kws={'size': 9}, linewidths=0.5)
ax.set_title('Matriz de Correlação de Pearson\n(triângulo inferior)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('graficos/05_heatmap_correlacao.png', bbox_inches='tight')
plt.close()
print("  ✓ graficos/05_heatmap_correlacao.png")

# Gráfico de barras — correlações com o alvo
fig, ax = plt.subplots(figsize=(10, 6))
cores_corr = ['#C44E52' if v > 0 else '#4C72B0' for v in corr_com_alvo.values]
ax.barh(corr_com_alvo.index, corr_com_alvo.values, color=cores_corr, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Correlação de Pearson com Doença Cardiovascular', fontsize=13, fontweight='bold')
ax.set_xlabel('Coeficiente de correlação')
ax.grid(True, alpha=0.3, axis='x')
for i, val in enumerate(corr_com_alvo.values):
    ax.text(val + (0.002 if val >= 0 else -0.002), i,
            f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=9)
plt.tight_layout()
plt.savefig('graficos/06_correlacao_com_alvo.png', bbox_inches='tight')
plt.close()
print("  ✓ graficos/06_correlacao_com_alvo.png")


# =============================================================================
# 7. IDENTIFICAÇÃO DE OUTLIERS E INCONSISTÊNCIAS
# =============================================================================
print("\n[7/8] Identificando outliers e inconsistências...")

print("\n  ANÁLISE DE OUTLIERS (método IQR):")
outlier_resumo = {}
for col, nome in [('age_years', 'Idade (anos)'), ('height', 'Altura'),
                   ('weight', 'Peso'), ('ap_hi', 'Pressão Sistólica'),
                   ('ap_lo', 'Pressão Diastólica'), ('bmi', 'IMC')]:
    Q1 = df_plot[col].quantile(0.25)
    Q3 = df_plot[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df_plot[(df_plot[col] < lower) | (df_plot[col] > upper)]
    pct = len(outliers) / len(df) * 100
    outlier_resumo[nome] = {'N outliers': len(outliers), '% outliers': round(pct, 2),
                             'Limite inferior': round(lower, 1), 'Limite superior': round(upper, 1),
                             'Min real': round(df_plot[col].min(), 1), 'Max real': round(df_plot[col].max(), 1)}
    print(f"  {nome:25s}: {len(outliers):5,} outliers ({pct:.2f}%)  |  "
          f"Limite [{lower:.1f}, {upper:.1f}]  |  Real [{df_plot[col].min():.1f}, {df_plot[col].max():.1f}]")

print("\n  VALORES CLINICAMENTE INVÁLIDOS (pressão arterial):")
invalidos_ap_hi = df[(df['ap_hi'] <= 0) | (df['ap_hi'] > 300)]
invalidos_ap_lo = df[(df['ap_lo'] <= 0) | (df['ap_lo'] > 200)]
invalidos_ap_neg = df[df['ap_hi'] < df['ap_lo']]

print(f"  Pressão sistólica ≤ 0 ou > 300 mmHg : {len(invalidos_ap_hi):,} registros")
print(f"  Pressão diastólica ≤ 0 ou > 200 mmHg: {len(invalidos_ap_lo):,} registros")
print(f"  Sistólica < Diastólica (impossível)  : {len(invalidos_ap_neg):,} registros")

invalidos_height = df[(df['height'] < 100) | (df['height'] > 220)]
invalidos_weight = df[(df['weight'] < 30) | (df['weight'] > 200)]
print(f"  Altura < 100 cm ou > 220 cm          : {len(invalidos_height):,} registros")
print(f"  Peso < 30 kg ou > 200 kg             : {len(invalidos_weight):,} registros")

total_invalidos = len(
    df[(df['ap_hi'] <= 0) | (df['ap_hi'] > 300) |
       (df['ap_lo'] <= 0) | (df['ap_lo'] > 200) |
       (df['ap_hi'] < df['ap_lo']) |
       (df['height'] < 100) | (df['height'] > 220) |
       (df['weight'] < 30) | (df['weight'] > 200)]
)
print(f"\n  Total de registros com ao menos 1 valor inválido: {total_invalidos:,} "
      f"({total_invalidos/len(df)*100:.2f}%)")

# Gráfico scatter pressão sistólica × diastólica com destaque dos inválidos
fig, ax = plt.subplots(figsize=(10, 7))
validos = df[(df['ap_hi'] > 0) & (df['ap_hi'] <= 300) &
             (df['ap_lo'] > 0) & (df['ap_lo'] <= 200) &
             (df['ap_hi'] >= df['ap_lo'])]
inval = df[~df.index.isin(validos.index)]

ax.scatter(validos['ap_lo'], validos['ap_hi'], alpha=0.1, s=5, color='#4C72B0', label='Válidos')
ax.scatter(inval['ap_lo'], inval['ap_hi'], alpha=0.6, s=10, color='#C44E52', label=f'Inválidos ({len(inval):,})')
ax.plot([0, 200], [0, 200], 'k--', linewidth=1, label='Diastólica = Sistólica')
ax.set_xlabel('Pressão Diastólica (mmHg)')
ax.set_ylabel('Pressão Sistólica (mmHg)')
ax.set_title('Pressão Arterial: Sistólica × Diastólica\n(destaque de valores inválidos)',
             fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-50, 250)
ax.set_ylim(-50, 350)
plt.tight_layout()
plt.savefig('graficos/07_pressao_invalidos.png', bbox_inches='tight')
plt.close()
print("  ✓ graficos/07_pressao_invalidos.png")


# =============================================================================
# 8. ANÁLISES ADICIONAIS
# =============================================================================
print("\n[8/8] Análises adicionais...")

# Distribuição de idade por doença cardiovascular (histograma sobreposto)
fig, ax = plt.subplots(figsize=(10, 6))
for label, cor in [('Sem doença', '#4C72B0'), ('Com doença', '#C44E52')]:
    dados = df_plot[df_plot['cardio_label'] == label]['age_years']
    ax.hist(dados, bins=30, alpha=0.5, color=cor, label=label,
            density=True, edgecolor='white')
    dados.plot.kde(ax=ax, color=cor, linewidth=2)
ax.set_title('Distribuição de Idade por Doença Cardiovascular', fontsize=13, fontweight='bold')
ax.set_xlabel('Idade (anos)')
ax.set_ylabel('Densidade')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graficos/08_idade_por_cardio.png', bbox_inches='tight')
plt.close()
print("  ✓ graficos/08_idade_por_cardio.png")

# Prevalência de doença por faixa etária
df_plot['faixa_etaria'] = pd.cut(df_plot['age_years'],
    bins=[29, 39, 44, 49, 54, 59, 65],
    labels=['30–39', '40–44', '45–49', '50–54', '55–59', '60–65'])

prev_faixa = df_plot.groupby('faixa_etaria', observed=True)['cardio'].mean() * 100

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(prev_faixa.index.astype(str), prev_faixa.values,
              color=sns.color_palette("Reds", len(prev_faixa)), edgecolor='white')
ax.set_title('Prevalência de Doença Cardiovascular por Faixa Etária', fontsize=13, fontweight='bold')
ax.set_xlabel('Faixa etária')
ax.set_ylabel('Prevalência (%)')
ax.set_ylim(0, 100)
for bar, val in zip(bars, prev_faixa.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
            f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('graficos/09_prevalencia_faixa_etaria.png', bbox_inches='tight')
plt.close()
print("  ✓ graficos/09_prevalencia_faixa_etaria.png")

# Pair plot das principais variáveis numéricas (amostra de 2000 para performance)
sample = df_plot[['age_years', 'bmi', 'ap_hi', 'ap_lo', 'cardio_label']].dropna()
sample = sample[(sample['ap_hi'] > 60) & (sample['ap_hi'] < 200) &
                (sample['ap_lo'] > 40) & (sample['ap_lo'] < 150) &
                (sample['bmi'] > 15) & (sample['bmi'] < 50)].sample(2000, random_state=42)

g = sns.pairplot(sample, hue='cardio_label', palette={'Sem doença': '#4C72B0', 'Com doença': '#C44E52'},
                 plot_kws={'alpha': 0.3, 's': 15},
                 vars=['age_years', 'bmi', 'ap_hi', 'ap_lo'],
                 diag_kind='kde')
g.fig.suptitle('Pair Plot — Variáveis Numéricas Principais\n(amostra de 2.000 registros)',
               y=1.02, fontsize=13, fontweight='bold')
g.fig.savefig('graficos/10_pairplot.png', bbox_inches='tight')
plt.close()
print("  ✓ graficos/10_pairplot.png")


# =============================================================================
# RESUMO FINAL
# =============================================================================
print("\n" + "=" * 65)
print("  RESUMO DOS ACHADOS DA EDA")
print("=" * 65)

print(f"""
  DATASET
  ├── Total de registros : {len(df):,}
  ├── Total de atributos : {df.shape[1]}
  ├── Valores nulos      : {df.isnull().sum().sum()} (dataset completo)
  └── Duplicatas         : {df.duplicated().sum()}

  VARIÁVEL ALVO
  ├── Sem doença cardiovascular : {(df['cardio']==0).sum():,} ({(df['cardio']==0).mean()*100:.1f}%)
  └── Com doença cardiovascular : {(df['cardio']==1).sum():,} ({(df['cardio']==1).mean()*100:.1f}%)
  → Dataset razoavelmente balanceado ✓

  PERFIL DOS PACIENTES
  ├── Idade média         : {df_plot['age_years'].mean():.1f} anos
  │   (mín: {df_plot['age_years'].min():.0f} | máx: {df_plot['age_years'].max():.0f})
  ├── IMC médio           : {df_plot['bmi'].mean():.1f} kg/m²
  ├── Pressão sistólica   : média {df['ap_hi'].mean():.0f} mmHg
  └── Pressão diastólica  : média {df['ap_lo'].mean():.0f} mmHg

  VARIÁVEIS MAIS CORRELACIONADAS COM DOENÇA (Pearson):
""")
for var, val in corr_com_alvo.head(5).items():
    print(f"    {var:22s}: {val:+.4f}")

print(f"""
  QUALIDADE DOS DADOS
  └── Registros com valores clinicamente inválidos: {total_invalidos:,}
      ({total_invalidos/len(df)*100:.2f}% do total)
      → Serão tratados na Etapa de Engenharia de Dados

  GRÁFICOS GERADOS (pasta 'graficos/'):
  01 — Histogramas com KDE das variáveis numéricas
  02 — Box plots por presença de doença
  03 — Distribuição da variável alvo
  04 — Variáveis categóricas × doença (barras empilhadas %)
  05 — Heatmap de correlação de Pearson
  06 — Correlação de cada variável com o alvo
  07 — Scatter pressão sistólica × diastólica (inválidos destacados)
  08 — Distribuição de idade por grupo (doença/sem doença)
  09 — Prevalência de doença por faixa etária
  10 — Pair plot das variáveis numéricas principais
""")
print("=" * 65)
print("  EDA concluída com sucesso!")
print("=" * 65)
