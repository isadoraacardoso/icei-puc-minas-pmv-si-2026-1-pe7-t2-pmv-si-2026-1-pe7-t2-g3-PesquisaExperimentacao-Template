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

Inicialmente, a inspeção da base revelou um conjunto robusto de **70.000 registros sem valores nulos**. A análise da distribuição da variável alvo demonstrou que o dataset é perfeitamente balanceado, contendo proporções equivalentes de pacientes saudáveis e com doença cardiovascular, o que configura um cenário ideal para a futura modelagem preditiva.

### Distribuição da Variável Alvo

![Distribuição da Variável Alvo](/src/graficos/03_distribuicao_variavel_alvo.png)

**Figura 03 - Distribuição da Variável Alvo:** O gráfico evidencia o balanceamento perfeito do dataset com aproximadamente 50% de observações em cada classe (presença/ausência de doença cardiovascular). Essa proporção equivalente é fundamental para evitar vieses na modelagem preditiva posterior.

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

**Figura 01 - Distribuição das Variáveis Contínuas:** Os histogramas com curva KDE revelam a distribuição de frequência das principais variáveis contínuas. A compressão visual nos quartis centrais é evidente, demonstrando a presença de outliers que afetam significativamente a visualização das distribuições normais.

### Box Plots por Presença de Doença

![Box Plots por Cardio](/src/graficos/02_boxplots_por_cardio.png)

**Figura 02 - Box Plots:** Os box plots segregam as distribuições por presença (1) e ausência (0) de doença cardiovascular. Os pontos dispersos além dos "bigodes" representam os outliers, cuja magnitude revela a necessidade urgente de tratamento desses valores anômalos.

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

Esse achado é fundamental, pois **direciona a modelagem preditiva da próxima etapa para a necessidade de utilização de algoritmos não-lineares mais complexos** (como baseados em árvores ou redes neurais).

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

## 📝 Observações Finais

A precisão dos futuros modelos preditivos dependerá fundamentalmente de um **pipeline de engenharia focado na correção das anomalias de digitação** descobertas nesta exploração. O balanceamento natural do dataset e a robustez estrutural observada sugerem excelentes perspectivas para a construção de modelos de classificação eficazes após o tratamento adequado dos problemas de qualidade identificados.

---

**Data de Análise:** Março de 2026  
**Versão:** 1.0  
**Status:** Análise Inicial Concluída

---

## 📚 Referências

- **Dataset:** Cardiovascular Disease Dataset - Kaggle
- **Fonte:** https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
- **Etapa do Projeto:** Análise Exploratória de Dados (EDA)
