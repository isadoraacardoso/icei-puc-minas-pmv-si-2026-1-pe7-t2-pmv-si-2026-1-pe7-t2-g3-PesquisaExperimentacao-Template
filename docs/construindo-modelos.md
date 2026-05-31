# Preparação dos Dados

Nesta etapa foi utilizada a base de dados `cardio_train_sem_valores_invalidos.csv`, gerada após o processo de limpeza realizado durante a Análise Exploratória de Dados (EDA). A etapa exploratória identificou diversos registros clinicamente inconsistentes, incluindo valores impossíveis para pressão arterial, altura e peso, que foram removidos antes da modelagem.

A base final utilizada na modelagem contém 68.677 registros e 15 colunas, preservando aproximadamente 98% dos dados originais e mantendo a representatividade do problema.

## Limpeza dos Dados

A limpeza teve como objetivo eliminar registros incompatíveis com limites fisiológicos humanos. Foram removidos registros que apresentavam:

* Pressão sistólica (`ap_hi`) menor ou igual a 0 ou superior a 300 mmHg;
* Pressão diastólica (`ap_lo`) menor ou igual a 0 ou superior a 200 mmHg;
* Pressão sistólica menor que a pressão diastólica;
* Altura inferior a 100 cm ou superior a 220 cm;
* Peso inferior a 30 kg ou superior a 200 kg.

Também foi realizada a verificação da existência de valores nulos, não sendo identificada a necessidade de técnicas de imputação.

Os outliers estatísticos identificados durante a análise exploratória não foram removidos automaticamente por métodos puramente estatísticos, como IQR, pois alguns desses valores poderiam representar pacientes reais em condições clínicas extremas. Dessa forma, optou-se por remover apenas registros considerados biologicamente impossíveis.

## Transformação dos Dados

A coluna `id` foi removida por não possuir valor preditivo para o problema.

A variável `age`, originalmente armazenada em dias, foi convertida para anos através da divisão por 365,25, gerando a variável `age_years`.

As variáveis contínuas utilizadas foram:

* age_years
* height
* weight
* ap_hi
* ap_lo
* IMC

Essas variáveis foram padronizadas utilizando o algoritmo StandardScaler, transformando os atributos para média zero e desvio padrão unitário.

A padronização foi especialmente importante para os experimentos com Support Vector Machine (SVM), uma vez que esse algoritmo é sensível à escala dos atributos.

As variáveis categóricas e binárias já estavam codificadas numericamente e foram mantidas sem transformações adicionais:

* gender
* cholesterol
* gluc
* smoke
* alco
* active

## Feature Engineering

Foi criada a variável IMC (Índice de Massa Corporal), calculada pela fórmula:

IMC = peso / (altura em metros)²

Essa variável foi incluída por sua relevância clínica, uma vez que o excesso de peso é reconhecido como fator associado ao desenvolvimento de doenças cardiovasculares.

## Tratamento de Dados Desbalanceados

Após a limpeza dos registros inválidos, a variável alvo permaneceu praticamente balanceada:

* Sem doença cardiovascular: 34.700 registros, aproximadamente 50,5%
* Com doença cardiovascular: 33.977 registros, aproximadamente 49,5%

Devido a esse equilíbrio natural, não foram aplicadas técnicas de oversampling ou undersampling.

Entretanto, foi avaliada uma configuração específica do algoritmo SVM utilizando o parâmetro `class_weight='balanced'`, permitindo verificar se o ajuste automático dos pesos das classes produziria ganhos de desempenho.

## Separação dos Dados

Os dados foram divididos em:

* 80% para treinamento;
* 20% para teste.

A divisão foi realizada de forma estratificada para preservar a distribuição original da variável alvo em ambos os conjuntos.

## Validação Cruzada

Além da divisão treino/teste, foi utilizada validação cruzada estratificada com cinco folds, permitindo avaliar a estabilidade dos modelos em diferentes particionamentos da base.

# Descrição dos Modelos

Nesta etapa foram implementados e avaliados os algoritmos Regressão Logística e Support Vector Machine (SVM), sendo seus resultados comparados ao modelo Random Forest desenvolvido na etapa anterior. O objetivo foi analisar diferentes abordagens de classificação para a predição de doença cardiovascular e identificar o modelo mais adequado ao problema.

## Regressão Logística

A Regressão Logística é um algoritmo de aprendizado supervisionado amplamente utilizado em problemas de classificação binária. Apesar do nome, trata-se de um modelo de classificação que estima a probabilidade de ocorrência de um evento por meio da função logística (sigmoide).

No contexto deste projeto, a Regressão Logística foi utilizada para prever a presença ou ausência de doença cardiovascular, representada pela variável alvo cardio.

Uma das principais vantagens desse algoritmo é sua simplicidade e interpretabilidade, permitindo compreender a influência dos atributos sobre a probabilidade de ocorrência da doença. Além disso, a Regressão Logística apresenta baixo custo computacional e costuma ser utilizada como modelo de referência (baseline) em problemas de classificação.

Entre suas principais vantagens destacam-se:

* Facilidade de implementação e interpretação;
* Baixo custo computacional;
* Boa capacidade de generalização;
* Produção de probabilidades associadas às previsões;
* Ampla utilização como modelo de comparação em problemas de classificação.

Entre suas limitações destacam-se:

* Assume uma relação linear entre os atributos e o logaritmo das chances da classe positiva;
* Pode apresentar desempenho inferior quando existem relações altamente não lineares entre as variáveis;
* É sensível à multicolinearidade entre atributos.

Configuração Utilizada

O modelo foi treinado utilizando os seguintes parâmetros:

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='lbfgs',
    n_jobs=-1
)

Os parâmetros foram definidos com o objetivo de garantir a convergência do algoritmo e a reprodutibilidade dos resultados.

Resultados Obtidos

A Regressão Logística apresentou os seguintes resultados na base de teste:

Métrica	Valor
AUC-ROC	0,7840
Acurácia	0,7207
Recall	0,6617

Os resultados demonstram que a Regressão Logística apresentou desempenho competitivo para o problema estudado, servindo como importante referência para comparação com modelos mais complexos, como Random Forest e Support Vector Machine.

## Support Vector Machine (SVM)

O Support Vector Machine é um algoritmo supervisionado amplamente utilizado em tarefas de classificação. Seu objetivo consiste em encontrar um hiperplano que maximize a separação entre classes distintas, buscando a maior margem possível entre os exemplos de cada grupo.

Uma das principais características do SVM é a utilização de funções kernel, que permitem representar relações não lineares entre as variáveis através da projeção dos dados para espaços de maior dimensionalidade.

Entre suas principais vantagens destacam-se:

* Boa capacidade de generalização;
* Eficiência em problemas de classificação binária;
* Capacidade de modelar relações lineares e não lineares;
* Robustez em bases com múltiplas variáveis.

Entre suas limitações destacam-se:

* Sensibilidade à escolha dos hiperparâmetros;
* Maior custo computacional em bases extensas;
* Necessidade de padronização dos dados.

## Experimentos Realizados

Foram avaliadas diferentes configurações do algoritmo:

### SVM RBF Padrão

Configuração padrão do kernel Radial Basis Function (RBF), amplamente utilizada para modelagem de relações não lineares.

### SVM RBF com C = 10

Aumentou-se o parâmetro C para tornar o modelo menos tolerante a erros de classificação durante o treinamento.

### SVM RBF com Gamma = 0,1

Foi avaliado um valor específico de gamma para alterar o alcance de influência dos exemplos de treinamento.

### SVM Linear

Modelo utilizando kernel linear, permitindo avaliar se uma fronteira linear seria suficiente para separar as classes.

### SVM Polinomial Grau 3

Modelo baseado em kernel polinomial de terceiro grau para capturar relações mais complexas.

### SVM Balanceado

Modelo utilizando:

class_weight = "balanced"

Essa configuração ajusta automaticamente os pesos das classes durante o treinamento.

# Avaliação dos Modelos Criados

## Métricas Utilizadas

Foram utilizadas as seguintes métricas:

### Acurácia

Representa a proporção total de classificações corretas realizadas pelo modelo.

### Precisão

Mede a proporção de previsões positivas que realmente pertencem à classe positiva.

### Recall

Representa a capacidade do modelo em identificar corretamente pacientes que possuem doença cardiovascular.

No contexto deste projeto, o Recall foi considerado a métrica prioritária, pois falsos negativos representam pacientes doentes classificados incorretamente como saudáveis.

### F1-Score

Corresponde à média harmônica entre precisão e recall.

### AUC-ROC

Avalia a capacidade de separação entre as classes em diferentes limiares de decisão.

## Resultados dos Modelos SVM

| Modelo                | AUC-ROC | Acurácia | Recall |
| --------------------- | ------- | -------- | ------ |
| SVM RBF Padrão        | 0,7943  | 0,7369   | 0,6848 |
| SVM RBF C=10          | 0,7887  | 0,7324   | 0,6792 |
| SVM RBF Gamma=0,1     | 0,7926  | 0,7357   | 0,6861 |
| SVM Linear            | 0,7958  | 0,7263   | 0,6325 |
| SVM Polinomial Grau 3 | 0,7831  | 0,7201   | 0,6096 |
| SVM Balanceado        | 0,7943  | 0,7365   | 0,6909 |

Observa-se que o SVM Balanceado apresentou o maior Recall entre as configurações avaliadas, enquanto o SVM Linear apresentou a maior AUC-ROC.

# Discussão dos Resultados Obtidos

A comparação entre os modelos desenvolvidos ao longo do projeto é apresentada a seguir.

| Modelo              | AUC-ROC | Acurácia | Recall |
| ------------------- | ------- | -------- | ------ |
| Logistic Regression | 0,7840  | 0,7207   | 0,6617 |
| Random Forest       | 0,7700  | 0,7080   | 0,6972 |
| SVM Balanceado      | 0,7943  | 0,7365   | 0,6909 |

Os resultados demonstram que o algoritmo SVM apresentou os melhores valores de AUC-ROC e acurácia, indicando maior capacidade de discriminação entre pacientes com e sem doença cardiovascular.

Entretanto, o objetivo principal deste estudo é maximizar a identificação correta dos pacientes com doença cardiovascular. Por esse motivo, o Recall foi definido como métrica prioritária.

Nesse critério, o Random Forest apresentou desempenho superior, alcançando Recall de aproximadamente 69,72%, enquanto o melhor modelo SVM obteve Recall de aproximadamente 69,09%.

Embora a diferença seja pequena, o Random Forest permaneceu ligeiramente superior na métrica mais importante para o contexto clínico do problema.

Assim, os resultados indicam que:

* O SVM apresentou melhor desempenho global em termos de discriminação das classes;
* O Random Forest apresentou maior capacidade de identificação de pacientes positivos;
* Embora a Regressão Logística tenha apresentado desempenho inferior ao SVM, seus resultados permaneceram competitivos e superiores ao Random Forest em termos de AUC-ROC, demonstrando que modelos lineares também podem ser eficazes para o problema estudado.

Dessa forma, considerando o objetivo do estudo e a métrica prioritária adotada, o modelo Random Forest foi mantido como modelo final do projeto.

# Revisão do Pipeline de Pesquisa e Análise de Dados

Após a realização das etapas de exploração, preparação, modelagem e avaliação, o pipeline inicialmente proposto foi revisado para torná-lo mais abrangente e aplicável a diferentes projetos de aprendizado de máquina.

O pipeline revisado é composto pelas seguintes etapas:

1. Definição do problema e dos objetivos da pesquisa;
2. Coleta dos dados;
3. Compreensão do domínio do problema;
4. Limpeza e preparação dos dados;
5. Análise exploratória dos dados;
6. Engenharia e seleção de atributos;
7. Tratamento de desbalanceamento de classes;
8. Divisão dos dados em conjuntos de treinamento e teste;
9. Validação cruzada para avaliação da estabilidade dos modelos;
10. Transformação dos dados (padronização ou normalização);
11. Implementação dos modelos candidatos;
12. Avaliação de diferentes configurações e hiperparâmetros dos modelos;
13. Validação cruzada;
14. Treinamento e comparação dos modelos;
15. Avaliação utilizando métricas adequadas ao problema;
16. Comparação dos modelos desenvolvidos;
17. Seleção do modelo final;
18. Documentação e reprodutibilidade dos experimentos;
19. Implantação e monitoramento contínuo.

As principais melhorias em relação ao pipeline original foram a inclusão explícita das etapas de validação cruzada, ajuste de hiperparâmetros, comparação sistemática de algoritmos e monitoramento contínuo, tornando o processo mais robusto e alinhado às boas práticas de ciência de dados e aprendizado de máquina.

# Referências

BREIMAN, Leo. Random forests. *Machine Learning*, v. 45, n. 1, p. 5–32, 2001.

CORTES, Corinna; VAPNIK, Vladimir. Support-vector networks. *Machine Learning*, v. 20, n. 3, p. 273–297, 1995.

PEDREGOSA, Fabian et al. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, v. 12, p. 2825–2830, 2011. Disponível em: https://scikit-learn.org. Acesso em: 31 maio 2026.

SULIANOVA, Svetlana. *Cardiovascular Disease Dataset*. Kaggle, 2019. Disponível em: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset. Acesso em: 7 mar. 2026.

HASTIE, Trevor; TIBSHIRANI, Robert; FRIEDMAN, Jerome. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2. ed. New York: Springer, 2009.

JAMES, Gareth et al. *An Introduction to Statistical Learning: with Applications in R*. 2. ed. New York: Springer, 2021.

MCKINNEY, Wes. Data structures for statistical computing in Python. In: *Proceedings of the 9th Python in Science Conference*. Austin: SciPy, 2010. p. 56–61. Disponível em: https://pandas.pydata.org. Acesso em: 20 mar. 2026.

HARRIS, Charles R. et al. Array programming with NumPy. *Nature*, v. 585, p. 357–362, 2020. Disponível em: https://numpy.org. Acesso em: 26 mar. 2026.

VIRTANEN, Pauli et al. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, v. 17, p. 261–272, 2020. Disponível em: https://scipy.org. Acesso em: 27 mar. 2026.

PYTHON SOFTWARE FOUNDATION. *Python Language Reference, version 3.x*. Wilmington: Python Software Foundation, 2024. Disponível em: https://www.python.org. Acesso em: 20 mar. 2026.

