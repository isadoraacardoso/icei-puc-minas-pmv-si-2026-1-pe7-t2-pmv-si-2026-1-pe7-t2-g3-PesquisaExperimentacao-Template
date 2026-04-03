import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =========================
# Criar pasta para gráficos
# =========================
os.makedirs('graficos', exist_ok=True)

# =========================
# Carregar base
# =========================
df = pd.read_csv('cardio_train_final.csv', sep=';')

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

print("\nEDA concluída com sucesso!")
print("Gráficos salvos na pasta: graficos/")