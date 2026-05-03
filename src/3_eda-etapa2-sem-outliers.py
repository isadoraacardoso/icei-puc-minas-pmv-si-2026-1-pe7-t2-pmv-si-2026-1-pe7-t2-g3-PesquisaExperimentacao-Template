import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import chi2_contingency
from pathlib import Path

# =========================
# Criar pasta para gráficos
# =========================
os.makedirs('graficos', exist_ok=True)

# =========================
# Carregar base (DADOS LIMPOS)
# =========================

BASE_DIR = Path(__file__).resolve().parent

ARQUIVO_ENTRADA = BASE_DIR / "cardio_train_sem_valores_invalidos.csv"

if not ARQUIVO_ENTRADA.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {ARQUIVO_ENTRADA}")

df = pd.read_csv(ARQUIVO_ENTRADA, sep=";")


# =========================
# Informações gerais
# =========================
print("Dimensões da base:")
print(df.shape)

print("\nInformações gerais:")
print(df.info())

print("\nEstatísticas descritivas:")
print(df.describe())

# =========================
# Histograma geral
# =========================
df.hist(figsize=(15,10))
plt.tight_layout()
plt.savefig('graficos/histogramas.png')
plt.show()

# =========================
# Heatmap de correlação
# =========================
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Mapa de Correlação')
plt.savefig('graficos/heatmap_correlacao.png')
plt.show()
# =========================
# Spearman (NOVO)
# =========================
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(method='spearman'), annot=True, cmap='coolwarm')
plt.title('Mapa de Correlação - Spearman')
plt.savefig('graficos/heatmap_spearman.png')
plt.show()

print("\nCorrelação de Spearman com a variável alvo:")
print(df.corr(method='spearman')['cardio'].sort_values(ascending=False))
# =========================
# Boxplot pressão sistólica
# =========================
plt.figure(figsize=(8,5))
sns.boxplot(x=df['ap_hi'])
plt.title('Boxplot Pressão Sistólica')
plt.savefig('graficos/boxplot_pressao.png')
plt.show()

# =========================
# Boxplot peso
# =========================
plt.figure(figsize=(8,5))
sns.boxplot(x=df['weight'])
plt.title('Boxplot Peso')
plt.savefig('graficos/boxplot_peso.png')
plt.show()

# =========================
# Colesterol x Cardio
# =========================
plt.figure(figsize=(8,5))
sns.countplot(x='cholesterol', hue='cardio', data=df)
plt.title('Colesterol x Doença Cardíaca')
plt.savefig('graficos/colesterol_cardio.png')
plt.show()

# =========================
# Glicose x Cardio
# =========================
plt.figure(figsize=(8,5))
sns.countplot(x='gluc', hue='cardio', data=df)
plt.title('Glicose x Doença Cardíaca')
plt.savefig('graficos/glicose_cardio.png')
plt.show()

# =========================
# Fumante x Cardio
# =========================
plt.figure(figsize=(8,5))
sns.countplot(x='smoke', hue='cardio', data=df)
plt.title('Fumante x Doença Cardíaca')
plt.savefig('graficos/fumante_cardio.png')
plt.show()
# =========================
# 🔴 Qui-quadrado + Cramér's V (NOVO)
# =========================

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(k-1, r-1))))

variaveis_categoricas = ['cholesterol', 'gluc', 'smoke', 'alco', 'active']

resultados = []

print("\n=== Teste Qui-Quadrado e Cramér's V ===")

for var in variaveis_categoricas:
    tabela = pd.crosstab(df[var], df['cardio'])
    
    chi2, p, dof, expected = chi2_contingency(tabela)
    v = cramers_v(tabela)
    
    print(f"\nVariável: {var}")
    print(f"Chi2: {chi2:.4f}")
    print(f"p-valor: {p:.6f}")
    print(f"Cramér's V: {v:.4f}")
    
    resultados.append([var, chi2, p, v])
print("\nEDA concluída com sucesso!")
print("Gráficos salvos na pasta: graficos/")
