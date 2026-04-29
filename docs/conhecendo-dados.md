# Análise Exploratória de Dados (EDA)
## Predição de Doença Cardiovascular

---

## 📋 Resumo Executivo

A presente etapa apresenta uma **Análise Exploratória de Dados (EDA)** abrangente sobre o dataset de predição de doenças cardiovasculares. O objetivo principal é compreender a estrutura do dataset, identificar valores atípicos (outliers) e mapear as relações numéricas e categóricas com a variável alvo.

**Destaques iniciais:**
- ✅ 70.000 registros sem valores nulos
- ✅ Dataset perfeitamente balanceado (50% saudáveis / 50% com doença)
- ⚠️ Presença de anomalias de qualidade requerendo tratamento rigoroso

---

## 1. Inspeção Inicial e Balanceamento

Inicialmente, a inspeção da base revelou um conjunto robusto de **70.000 registros sem valores nulos**. A análise da distribuição da variável alvo demonstrou que o dataset é aproximadamente balanceado, contendo proporções equivalentes de pacientes saudáveis e com doença cardiovascular, o que configura um cenário ideal para a futura modelagem preditiva.

### Distribuição da Variável Alvo

![Distribuição da Variável Alvo](/src/graficos/03_distribuicao_variavel_alvo.png)

**Figura 03 - Distribuição da Variável Alvo:** O gráfico evidencia o balanceamento do dataset com aproximadamente 50% de observações em cada classe (presença/ausência de doença cardiovascular). Essa proporção equivalente é fundamental para evitar vieses na modelagem preditiva posterior.

- Antes da limpeza:

Sem doença: ~50%
Com doença: ~50%

- Após a limpeza:

Sem doença: 51,03%
Com doença: 48,97%

Portanto, o dataset pode ser considerado aproximadamente balanceado, o que favorece a modelagem preditiva.

---

## 2. Engenharia Inicial de Atributos

Para enriquecer a análise e facilitar a interpretação clínica, foi realizada uma engenharia inicial de atributos:

- **Conversão de Idade:** A idade dos pacientes, originalmente registrada em dias, foi convertida para anos
- **Cálculo do IMC:** Calculou-se o Índice de Massa Corporal (IMC) a partir do peso e da altura
- **Perfil Médio da Amostra:** Pacientes com idade média de **53 anos** e IMC indicativo de **sobrepeso**

### Medidas de Tendência Central e Dispersão

| Métrica | Descrição |
|---------|-----------|
| **Desvio Padrão** | Elevado em variáveis contínuas como pressão arterial |
| **Intervalo Interquartil (IQR)** | Variabilidade anormal identificada |
| **Distribuição** | Extrema compressão visual nos quartis centrais |

---

## 3. Identificação de Outliers e Anomalias

As medidas de dispersão indicaram uma variabilidade anormal em colunas contínuas, como pressão arterial e altura. A visualização dessas distribuições evidenciou uma extrema compressão visual nos quartis centrais, causada por uma quantidade massiva de valores atípicos.

### Distribuição mediante Histogramas (KDE)

![Histogramas com Curva KDE](/src/graficos/01_histogramas_distribuicao.png)

**Figura 01 - Distribuição das Variáveis Contínuas:** Os histogramas com curva KDE revelam a distribuição de frequência das principais variáveis contínuas. (Nota: Recorte visual aplicado entre os quantis 1% e 99% para manter a legibilidade da escala). A compressão visual nos quartis centrais é evidente, demonstrando a presença de outliers que afetam significativamente a visualização das distribuições normais.

### Box Plots por Presença de Doença

![Box Plots por Cardio](/src/graficos/02_boxplots_por_cardio.png)

**Figura 02 - Box Plots:** Os box plots segregam as distribuições por presença (1) e ausência (0) de doença cardiovascular. (Nota: Recorte visual aplicado entre os quantis 1% e 99% para manter a legibilidade da escala). Os pontos dispersos além dos "bigodes" representam os outliers visíveis neste escopo. A magnitude real e absoluta dos valores anômalos extremos que extrapolam essa visualização (ex: pressões zeradas ou acima de 10.000) foi isolada e está detalhada no arquivo *(colocar o caminho do arquivo aqui Ex: tabelas/relatorio_outliers_extremos.csv).*

### Análise Aprofundada de Outliers

A investigação aprofundada desses outliers revelou um **problema crítico de qualidade nos dados**. Ao aplicar regras de negócio baseadas em limites biológicos reais, foram detectados registros clinicamente impossíveis, tais como:

- ❌ Pressões arteriais zeradas, negativas ou superiores a 300 mmHg
- ❌ Alturas extremas (como 70 centímetros em adultos)
- ❌ Casos anômalos com pressão diastólica superior à sistólica
- ❌ Erros de digitação e registros incompatíveis com a fisiologia humana

### Validação Biológica: Pressão Sistólica vs. Diastólica

![Pressão Sistólica vs. Diastólica - Valores Inválidos](/src/graficos/07_pressao_invalidos.png)

**Figura 07 - Detecção de Anomalias na Pressão Arterial:** O gráfico de dispersão que cruza a pressão sistólica e diastólica isola visualmente essas anomalias em vermelho, confirmando que uma parcela significativa dos dados sofreu erros de digitação. Registros com pressão diastólica superior à sistólica ou valores biologicamente impossíveis são evidentes, exigindo filtragem rigorosa na etapa de limpeza.

---

## 4. Análise Bivariada e Fatores de Risco

### Relação entre Categorias de Risco e Doença

![Categorias por Doença](/src/graficos/04_categoricas_por_cardio.png)

**Figura 04 - Fatores de Risco Categóricos:** O gráfico de barras empilhadas relaciona categorias de risco (colesterol e glicose) à presença da doença cardiovascular. A análise demonstra que níveis classificados como "Muito Acima do Normal" elevam consideravelmente a proporção de pacientes doentes, evidenciando forte associação entre esses marcadores e a presença da condição.

### Prevalência por Faixa Etária

![Prevalência por Faixa Etária](/src/graficos/09_prevalencia_faixa_etaria.png)

**Figura 09 - Doença por Faixa Etária:** O gráfico que demonstra a prevalência por faixa etária corrobora a correlação positiva de que o avanço da idade acompanha o aumento direto na incidência da doença cardiovascular. Observa-se uma tendência crescente clara conforme a idade avança.

### Análise Adicional: Idade como Fator de Risco

![Idade por Doença](/src/graficos/08_idade_por_cardio.png)

**Figura 08 - Distribuição de Idade:** Comparação da distribuição etária entre indivíduos com e sem doença. A sobreposição entre as distribuições indica que, embora a idade seja um fator de risco relevante, não é determinante isoladamente.

---

## 5. Análise de Correlação Linear

### Matriz de Correlação Geral

Para quantificar as relações lineares entre as variáveis contínuas, utilizou-se o **coeficiente de correlação de Pearson**.

![Heatmap de Correlação](/src/graficos/05_heatmap_correlacao.png)

**Figura 05 - Matriz de Correlação de Pearson:** O heatmap apresenta a correlação entre todas as variáveis contínuas do dataset. As cores mais intensas indicam correlações mais fortes. Observa-se que a pressão sistólica apresenta correlação substancial com a pressão diastólica e com a variável alvo.

### Correlação com a Variável Alvo

![Correlação com Alvo](/src/graficos/06_correlacao_com_alvo.png)

**Figura 06 - Correlação de Cada Variável com o Alvo:** O gráfico de barras ordena as variáveis pela magnitude de sua correlação linear com a presença de doença cardiovascular. Os resultados indicam que:

1. **Pressão Sistólica** → Preditor linear mais forte (r ≈ 0.42)
2. **Idade** → Segunda variável mais correlacionada (r ≈ 0.28)
3. **Colesterol** → Terceira em importância (r ≈ 0.22)

Em contrapartida, variáveis isoladas de hábitos de vida, como **tabagismo e consumo de álcool**, apresentaram correlação linear fraca neste contexto específico.

---

## 6. Análise Multivariada

### Pairplot de Variáveis Principais

![Pairplot Multivariado](/src/graficos/10_pairplot.png)

**Figura 10 - Relações Multivariadas:** A exploração multivariada através do pairplot cruzando Idade, IMC e Pressões demonstra a complexidade do problema estrutural subjacente. Ao colorir os pontos de dados pela presença ou ausência da doença, notou-se:

- ⚠️ **Forte sobreposição entre as classes**
- ⚠️ **Inexistência de separação linear simples**
- ✅ **Estrutura de dados revelando complexidade não-linear**

Esse achado é fundamental, pois **sugere a modelagem preditiva da próxima etapa para a de utilização de algoritmos não-lineares mais complexos** (como baseados em árvores ou redes neurais), mas não descarta modelos lineares como baseline.

---

## 7. Conclusões e Recomendações

### Potencial Preditivo
✅ O dataset possui **alto potencial preditivo** e **estrutura sólida**, com:
- Balanceamento perfeito de classes
- 70.000 observações robustas
- Variáveis com poder discriminativo comprovado

### Limitações Identificadas
⚠️ **Problemas de qualidade críticos:**
- Valores biologicamente impossíveis em pressão arterial
- Erros de digitação massivos em altura e pressões
- Necessidade de filtragem rigorosa de outliers

### Recomendações para Etapas Futuras

1. **Limpeza de Dados (Data Cleaning)**
   - Remover registros com pressão diastólica > sistólica
   - Filtrar valores de pressão fora do intervalo biológico (0-300 mmHg)
   - Validar e corrigir medidas de altura extremas

2. **Modelagem Preditiva**
   - Empregar algoritmos não-lineares (Random Forest, Gradient Boosting, Redes Neurais)
   - Evitar modelos lineares simples dada a complexidade revelada
   - Realizar validação cruzada rigorosa

3. **Feature Engineering Avançado**
   - Explorar interações entre pressão sistólica e diastólica
   - Criar bins etários mais refinados
   - Normalizar variáveis contínuas

---

## 📊 Resumo das Estatísticas Descritivas

As estatísticas descritivas detalhadas estão disponibilizadas em:
📄 [Tabela de Estatísticas Descritivas](/src/graficos/tabela_estatisticas_descritivas.csv)

---

## 📁 Estrutura de Arquivos

```
src/
├── cardio_train.csv                          # Dataset original
├── eda-etapa2.py                             # Script de análise
├── requirements.txt                          # Dependências
└── graficos/
    ├── 01_histogramas_distribuicao.png       # Análise de distribuições
    ├── 02_boxplots_por_cardio.png            # Outliers por classe
    ├── 03_distribuicao_variavel_alvo.png     # Balanceamento de classes
    ├── 04_categoricas_por_cardio.png         # Fatores de risco
    ├── 05_heatmap_correlacao.png             # Matriz de correlação
    ├── 06_correlacao_com_alvo.png            # Importância de variáveis
    ├── 07_pressao_invalidos.png              # Anomalias detectadas
    ├── 08_idade_por_cardio.png               # Idade como preditor
    ├── 09_prevalencia_faixa_etaria.png       # Prevalência etária
    ├── 10_pairplot.png                       # Análise multivariada
    └── tabela_estatisticas_descritivas.csv   # Estatísticas
```

---

## 🔬 Metodologia

- **Linguagem:** Python 3.x
- **Bibliotecas Principais:** Pandas, NumPy, Matplotlib, Seaborn, SciPy
- **Técnicas Aplicadas:**
  - Análise descritiva (média, mediana, desvio padrão)
  - Análise de outliers (IQR, visualização)
  - Correlação de Pearson
  - Visualizações exploratórias (histogramas, box plots, heatmaps, pairplots)

---

## 8. Reanálise Após Limpeza por Critérios Clínicos e Remoção de Valores Anômalos

Após a etapa inicial de identificação de anomalias, foi realizada uma nova análise exploratória utilizando a base já tratada (`cardio_train_sem_outliers.csv`). Essa segunda análise tem como objetivo verificar o comportamento estatístico das variáveis após a remoção dos registros biologicamente inconsistentes.

A nova base apresenta distribuições mais estáveis e permite interpretações estatísticas mais confiáveis, reduzindo o impacto de valores extremos anteriormente detectados.

---

### Carregamento da Base Tratada

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Criar pasta para armazenar os gráficos
os.makedirs('graficos', exist_ok=True)

# Carregar base já tratada
df = pd.read_csv('cardio_train_sem_outliers.csv', sep=';')
```

Nesta etapa, foi utilizada a base já limpa, contendo apenas registros considerados biologicamente plausíveis. A criação automática da pasta graficos garante organização dos arquivos gerados.

### Estatísticas Gerais da Base Pós-Limpeza

```
print("Dimensões da base:")
print(df.shape)

print("\nInformações gerais:")
print(df.info())

print("\nEstatísticas descritivas:")
print(df.describe())
Comentário do código
```

O comando shape permite verificar quantos registros permaneceram após a limpeza. Já describe() fornece medidas fundamentais como média, quartis e desvio padrão.

#### Após a limpeza:

- Redução da variabilidade extrema
- Quartis mais coerentes
- Desvio padrão reduzido em pressão arterial
- Melhor distribuição de peso e altura
- Permaneceram 66.488 registros válidos, correspondendo a aproximadamente 94,98% da base original. 

### Histogramas das Variáveis Após Limpeza

<img width="1500" height="589" alt="Figure_1" src="https://github.com/user-attachments/assets/054ff8e8-0757-4853-bf42-66319ee76485" /> <br>

```
df.hist(figsize=(15,10))
plt.tight_layout()
plt.savefig('graficos/histogramas.png')
plt.show()
```

Os histogramas permitem observar a distribuição de frequência de todas as variáveis numéricas simultaneamente.

#### Após a remoção dos outliers:

- As distribuições tornaram-se mais concentradas
- A pressão arterial perdeu caudas artificiais extremas
- Peso e altura apresentaram comportamento mais próximo do esperado biologicamente

### Heatmap de Correlação Atualizado

<img width="1200" height="636" alt="Figure_2" src="https://github.com/user-attachments/assets/bf0d825f-c6b7-4aea-8919-624ff8b1f5df" /> <br>

```
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Mapa de Correlação')
plt.savefig('graficos/heatmap_correlacao.png')
plt.show()
```

A matriz de correlação foi recalculada para avaliar se a remoção de outliers alterou relações lineares entre variáveis.

#### Mantiveram-se como principais relações:

- Pressão sistólica ↔ variável alvo
- Idade ↔ variável alvo
- Colesterol ↔ variável alvo

A limpeza reduziu ruídos estatísticos, tornando os coeficientes mais estáveis.

### Complemento da Análise de Correlação

Para aprimorar a análise estatística e considerar diferentes tipos de variáveis, foram aplicadas técnicas complementares à correlação de Pearson.

- Para variáveis ordinais (como colesterol e glicose), foi utilizada a **correlação de Spearman**, que captura relações monotônicas.
  
  <img width="1203" height="800" alt="Figure sperman" src="https://github.com/user-attachments/assets/c969ca47-1d17-4e74-80c3-e565666c429e" />
  
- Para variáveis categóricas (como tabagismo, consumo de álcool e atividade física), foi aplicado o **teste qui-quadrado de independência**. E para medir a intensidade da associação entre variáveis categóricas e a variável alvo, foi utilizado o **coeficiente de Cramér’s V**.

   | Variável           | p-valor  | Cramér’s V | Interpretação                  |
   |-------------------|---------|-----------|-------------------------------|
   | Colesterol        | < 0.001 | 0.2203    | Associação fraca a moderada   |
   | Glicose           | < 0.001 | 0.0909    | Associação muito fraca        |
   | Tabagismo         | < 0.001 | 0.0193    | Associação muito fraca        |
   | Álcool            | 0.0072  | 0.0104    | Associação muito fraca        |
   | Atividade Física  | < 0.001 | 0.0375    | Associação muito fraca        |

   Os resultados do teste qui-quadrado indicam que todas as variáveis analisadas apresentam associação estatisticamente significativa com a presença de doença cardiovascular (p < 0.05).

   No entanto, ao analisar a intensidade dessas associações por meio do coeficiente de Cramér’s V, observa-se que a maioria das variáveis apresenta associação muito fraca.

   A variável colesterol se destaca como a mais relevante entre as categóricas, apresentando associação de intensidade fraca a moderada com a variável alvo. Já glicose, tabagismo, consumo de álcool e atividade física         apresentam associação muito fraca quando consideradas isoladamente.

   Esses resultados indicam que, embora algumas variáveis comportamentais e clínicas tenham relação estatística com a doença, sua capacidade explicativa individual é limitada, reforçando a necessidade de análise              multivariada e uso de modelos preditivos que considerem a combinação de múltiplos fatores.

   Destaca-se que correlação ou associação estatística não implica causalidade.

### Boxplot da Pressão Sistólica

<img width="800" height="500" alt="Figure_3" src="https://github.com/user-attachments/assets/def5ace5-f7d7-40af-95d3-a2aea4531728" /> <br>

````
plt.figure(figsize=(8,5))
sns.boxplot(x=df['ap_hi'])
plt.title('Boxplot Pressão Sistólica')
plt.savefig('graficos/boxplot_pressao.png')
plt.show()
````

### Boxplot do Peso

<img width="800" height="500" alt="Figure_4" src="https://github.com/user-attachments/assets/0b25953c-7f99-43af-ad7d-7b3330f7f2e4" /> <br>

````
plt.figure(figsize=(8,5))
sns.boxplot(x=df['weight'])
plt.title('Boxplot Peso')
plt.savefig('graficos/boxplot_peso.png')
plt.show()
````

Os boxplots mostram claramente que, após a limpeza, ainda existem variações naturais, porém sem os extremos absurdos encontrados anteriormente.

### Relação entre Colesterol e Doença Cardiovascular

<img width="800" height="500" alt="Figure_5" src="https://github.com/user-attachments/assets/6a20c1c8-31a4-4d76-98b0-c142d9c78685" /> <br>

````
plt.figure(figsize=(8,5))
sns.countplot(x='cholesterol', hue='cardio', data=df)
plt.title('Colesterol x Doença Cardíaca')
plt.savefig('graficos/colesterol_cardio.png')
plt.show()
````

Pacientes com colesterol elevado continuam apresentando maior incidência de doença cardiovascular.

### Relação entre Glicose e Doença Cardiovascular

<img width="800" height="500" alt="Figure_6" src="https://github.com/user-attachments/assets/6c393f34-8a23-48e0-9a2b-ae9202324a1f" /> <br>

````
plt.figure(figsize=(8,5))
sns.countplot(x='gluc', hue='cardio', data=df)
plt.title('Glicose x Doença Cardíaca')
plt.savefig('graficos/glicose_cardio.png')
plt.show()
````

Níveis elevados de glicose mantêm associação importante com aumento da prevalência da doença.

### Relação entre Tabagismo e Doença Cardiovascular

<img width="800" height="500" alt="Figure_7" src="https://github.com/user-attachments/assets/dab246bd-a6c0-47a5-bb33-2392298e64c6" /> <br>

````
plt.figure(figsize=(8,5))
sns.countplot(x='smoke', hue='cardio', data=df)
plt.title('Fumante x Doença Cardíaca')
plt.savefig('graficos/fumante_cardio.png')
plt.show()
````

A variável tabagismo apresenta correlação visual mais fraca isoladamente, sugerindo efeito combinado com outros fatores.

### Conclusão Pós-Limpeza

A nova análise confirma que a remoção de outliers melhorou significativamente a qualidade estatística da base.

Principais ganhos:

- Distribuições mais realistas
- Correlações mais confiáveis
- Redução de ruído estatístico
- Melhor preparação para modelagem preditiva

A análise inicial foi mantida para demonstrar o impacto dos outliers, enquanto a análise subsequente utiliza a base tratada para evidenciar a melhoria estatística obtida após a limpeza.

## 📝 Observações Finais

A precisão dos futuros modelos preditivos dependerá fundamentalmente de um **pipeline de engenharia focado na correção das anomalias de digitação** descobertas nesta exploração. O balanceamento natural do dataset e a robustez estrutural observada sugerem excelentes perspectivas para a construção de modelos de classificação eficazes após o tratamento adequado dos problemas de qualidade identificados.

---

**Data de Análise:** Março de 2026  
**Versão:** 1.0  
**Status:** Análise Inicial Concluída

---

## 📚 Referências

SULIANOVA, Svetlana. Cardiovascular Disease dataset. Kaggle, 2019. Disponível em: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset . Acesso em: 7 mar. 2026.

EBAC ONLINE. Análise exploratória de dados (AED): o que é, ferramentas, técnicas e exemplos. Disponível em: Acessar artigo . Acesso em: 9 mar. 2026.

MEDRI, Waldir. Análise exploratória de dados. Londrina: Universidade Estadual de Londrina, 2011. Disponível em: https://docs.ufpr.br/~benitoag/apostilamedri.pdf . Acesso em: 11 mar. 2026.

SANDOVAL, Mônica Carneiro. Estatística descritiva. São Paulo: Universidade de São Paulo (USP), 2014. Disponível em: http://wiki.icmc.usp.br/images/2/23/Estat%C3%ADsticaDescritiva2014_1.pdf . Acesso em: 15 mar. 2026.

PYTHON SOFTWARE FOUNDATION. *Python Language Reference, version 3.x*. Wilmington: Python Software Foundation, 2024. Disponível em: <https://www.python.org>. Acesso em: 20 mar. 2026.

MCKINNEY, Wes. *Data structures for statistical computing in Python*. In: PYTHON IN SCIENCE CONFERENCE, 9., 2010, Austin. Proceedings [...]. Austin: SciPy, 2010. p. 56-61. Disponível em: <https://pandas.pydata.org>. Acesso em: 20 mar. 2026.

HARRIS, Charles R. et al. *Array programming with NumPy*. Nature, Londres, v. 585, p. 357–362, 2020. Disponível em: <https://numpy.org>. Acesso em: 26 mar. 2026.

HUNTER, John D. *Matplotlib: A 2D graphics environment*. Computing in Science & Engineering, v. 9, n. 3, p. 90-95, 2007. Disponível em: <https://matplotlib.org>. Acesso em: 26 mar. 2026.

WAASKOM, Michael L. *Seaborn: statistical data visualization*. Journal of Open Source Software, v. 6, n. 60, 2021. Disponível em: <https://seaborn.pydata.org>. Acesso em: 27 mar. 2026.

VIRTANEN, Pauli et al. *SciPy 1.0: fundamental algorithms for scientific computing in Python*. Nature Methods, v. 17, p. 261-272, 2020. Disponível em: <https://scipy.org>. Acesso em: 27 mar. 2026.
