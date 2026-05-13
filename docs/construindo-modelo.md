# Preparação dos dados

Nesta etapa, foi utilizada a base `cardio_train_sem_valores_invalidos.csv`, gerada a partir da remoção de registros clinicamente inválidos identificados na etapa anterior de análise exploratória. A base final utilizada na modelagem possui 68.677 registros e 15 colunas.

### Limpeza de dados

A limpeza dos dados foi conduzida com foco na remoção de valores clinicamente inválidos, ou seja, registros que apresentavam inconsistências incompatíveis com limites fisiológicos humanos. Foram considerados inválidos os seguintes casos:

- pressão sistólica (`ap_hi`) menor ou igual a 0 ou maior que 300 mmHg;
- pressão diastólica (`ap_lo`) menor ou igual a 0 ou maior que 200 mmHg;
- pressão sistólica menor que a pressão diastólica;
- altura menor que 100 cm ou maior que 220 cm;
- peso menor que 30 kg ou maior que 200 kg.

Na execução do pipeline da Etapa 3, nenhum novo registro foi removido, pois a base já havia passado anteriormente pela limpeza dos valores clinicamente inválidos. Também foi verificada a ausência de valores nulos, não sendo necessária nenhuma técnica de imputação.

Os outliers estatísticos identificados na etapa exploratória não foram removidos automaticamente pelo método IQR. Essa decisão foi tomada porque, no contexto de doença cardiovascular, valores extremos de pressão arterial, peso e IMC podem representar pacientes reais de maior risco clínico, e sua remoção poderia reduzir informações relevantes para o modelo.

### Transformação de dados

A coluna `id` foi removida por representar apenas um identificador do paciente, sem valor preditivo para o problema. A permanência dessa variável poderia induzir o modelo a aprender padrões artificiais sem significado clínico.

A variável `age`, originalmente representada em dias, foi convertida para anos por meio da divisão por 365,25. A nova variável foi renomeada para `age_years`, tornando a informação mais interpretável e adequada ao contexto clínico.

As variáveis contínuas utilizadas no modelo foram:

- `age_years`;
- `height`;
- `weight`;
- `ap_hi`;
- `ap_lo`;
- `IMC`.

Essas variáveis foram padronizadas com `StandardScaler`, que transforma os valores para uma escala com média 0 e desvio padrão 1. Embora o Random Forest não exija padronização, essa etapa foi mantida no pipeline para garantir maior padronização do processo e permitir reaproveitamento futuro com algoritmos sensíveis à escala, como Regressão Logística e SVM.

As variáveis categóricas e binárias já estavam codificadas numericamente no dataset original e foram mantidas sem transformação adicional:

- `gender`;
- `cholesterol`;
- `gluc`;
- `smoke`;
- `alco`;
- `active`.

### Feature Engineering

Foi criada a variável `IMC`, calculada a partir da relação entre peso e altura:

`IMC = peso / (altura em metros)²`

Essa variável foi incluída por sua relevância clínica, pois o índice de massa corporal pode estar associado ao risco cardiovascular e combina duas informações importantes da base: peso e altura.

### Tratamento de dados desbalanceados

A variável alvo `cardio` apresentou distribuição praticamente balanceada após a limpeza:

- sem doença cardiovascular: 34.700 registros, aproximadamente 50,5%;
- com doença cardiovascular: 33.977 registros, aproximadamente 49,5%.

Dessa forma, não foi necessário aplicar técnicas de balanceamento como oversampling ou undersampling. Para preservar essa distribuição nas etapas de treinamento e teste, foi utilizada divisão estratificada dos dados.

### Separação dos dados

Para a etapa final de treinamento e avaliação do modelo, foi utilizada a estratégia de separação treino/teste na proporção 80/20. Embora tenham sido discutidas alternativas de divisão como 75/25 e 70/30 durante a fase exploratória do projeto, apenas o experimento com divisão 80/20 foi efetivamente implementado e mantido na versão final do pipeline disponibilizada no repositório.

### Validação cruzada

Além das divisões treino/teste, foi aplicada validação cruzada estratificada com 5 folds, utilizando AUC-ROC como métrica de avaliação. Essa validação foi utilizada para verificar a estabilidade do modelo em diferentes partições da base.

# Descrição do modelo

O algoritmo selecionado para a construção do modelo foi o Random Forest Classifier.

O Random Forest é um algoritmo de aprendizado supervisionado baseado em ensemble, composto por múltiplas árvores de decisão. Cada árvore é treinada com uma amostra dos dados e realiza uma predição individual. Ao final, o modelo combina os resultados das árvores para produzir uma classificação final.

Neste projeto, o problema é de classificação binária, pois o objetivo é prever se um paciente possui ou não doença cardiovascular, representada pela variável alvo `cardio`.

### Justificativa da escolha

O Random Forest foi escolhido por ser adequado a problemas de classificação binária e por apresentar bom desempenho em bases com variáveis de diferentes tipos, como variáveis contínuas, ordinais e binárias. Além disso, o algoritmo consegue capturar relações não lineares entre os atributos e a variável alvo, o que é relevante no contexto clínico, em que fatores como idade, pressão arterial, colesterol, glicose e IMC podem interagir de forma complexa.

Outra vantagem do Random Forest é sua robustez em relação a ruídos e sua capacidade de fornecer a importância das variáveis, permitindo uma análise interpretativa sobre quais atributos mais contribuíram para a predição.

### Parâmetros utilizados

Foi utilizado um pipeline com duas etapas principais:

1. pré-processamento das variáveis;
2. treinamento do modelo Random Forest.

O pré-processamento foi implementado com `ColumnTransformer`, aplicando `StandardScaler` às variáveis contínuas e mantendo as variáveis categóricas/binárias sem alteração.

O modelo foi configurado com os seguintes parâmetros principais:

- `n_estimators=100`: número de árvores da floresta;
- `random_state=42`: garante reprodutibilidade dos resultados;
- `n_jobs=-1`: permite utilizar todos os núcleos disponíveis da máquina para acelerar o treinamento;
- `max_features`: estratégia de seleção de atributos.

Exemplo de configurações avaliadas:
| Configuração | Parâmetros                                | Resultado observado                                     |
| ------------ | ----------------------------------------- | ------------------------------------------------------- |
| Modelo 1     | `n_estimators=50`, `max_features="sqrt"`  | desempenho inferior e maior variação                    |
| Modelo 2     | `n_estimators=100`, `max_features="sqrt"` | melhor equilíbrio entre desempenho e tempo              |
| Modelo 3     | `n_estimators=200`, `max_features="log2"` | ganho pouco significativo com maior custo computacional |

Após os testes, optou-se pela configuração:

`RandomForestClassifier(
    n_estimators=100,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)`

A escolha foi baseada no melhor equilíbrio entre desempenho, estabilidade e custo computacional. O parâmetro max_features="sqrt" foi mantido por ser uma configuração frequentemente recomendada para Random Forest em problemas de classificação, contribuindo para maior diversidade entre as árvores e redução de overfitting. 

# Avaliação dos modelos criados

Foram avaliados três cenários de divisão treino/teste:

| Divisão | Treino | Teste | AUC-ROC | Acurácia | Precisão | Recall | F1-score |
|---|---:|---:|---:|---:|---:|---:|---:|
| 80/20 | 54.941 | 13.736 | 0,7700 | 0,7080 | 0,7081 | 0,6972 | 0,7026 |
| 75/25 | 51.507 | 17.170 | 0,7699 | 0,7105 | 0,7111 | 0,6988 | 0,7049 |
| 70/30 | 48.073 | 20.604 | 0,7710 | 0,7097 | 0,7101 | 0,6984 | 0,7042 |

Os resultados foram bastante próximos entre as três divisões, indicando estabilidade do modelo. A AUC-ROC permaneceu em torno de 0,77 em todos os cenários, enquanto a acurácia ficou próxima de 0,71.

A validação cruzada estratificada com 5 folds apresentou AUC-ROC média de 0,7713 e desvio padrão de 0,0032. O baixo desvio padrão indica que o modelo teve comportamento consistente em diferentes partições da base.

## Métricas utilizadas

As métricas utilizadas para avaliação do modelo foram:

### Acurácia

A acurácia mede a proporção total de classificações corretas realizadas pelo modelo. Como a base está relativamente balanceada entre pacientes com e sem doença cardiovascular, essa métrica é útil para uma visão geral do desempenho.

### Precisão

A precisão mede, entre todos os pacientes classificados pelo modelo como portadores de doença cardiovascular, quantos realmente pertenciam à classe positiva. Essa métrica é importante para avaliar a quantidade de falsos positivos.

### Recall

O recall, também chamado de sensibilidade, mede a capacidade do modelo de identificar corretamente os pacientes que realmente possuem doença cardiovascular. No contexto clínico, essa métrica é especialmente importante, pois falsos negativos representam pacientes doentes classificados como saudáveis.

### F1-score

O F1-score representa uma média harmônica entre precisão e recall. Essa métrica é útil para avaliar o equilíbrio entre a capacidade do modelo de evitar falsos positivos e falsos negativos.

### AUC-ROC

A AUC-ROC avalia a capacidade discriminativa do modelo em diferentes limiares de decisão, medindo o quanto o modelo consegue separar corretamente pacientes com e sem doença cardiovascular.

Embora a métrica AUC-ROC tenha sido utilizada para avaliar a capacidade geral de discriminação do modelo, o contexto clínico do problema exige atenção especial aos falsos negativos. Em aplicações relacionadas à saúde cardiovascular, deixar de identificar corretamente um paciente com risco da doença pode gerar consequências mais graves do que classificar incorretamente um paciente saudável como pertencente à classe positiva.

Dessa forma, além da análise da AUC-ROC, a sensibilidade (recall) foi considerada uma métrica prioritária na avaliação do modelo, pois mede a capacidade de identificar corretamente os pacientes positivos. Essa abordagem torna a avaliação mais coerente com o objetivo clínico do problema.

## Discussão dos resultados obtidos

O modelo Random Forest apresentou desempenho moderado e consistente na predição de doença cardiovascular. As três divisões treino/teste avaliadas apresentaram resultados muito próximos, com AUC-ROC próxima de 0,77 e acurácia próxima de 0,71.

A divisão 70/30 apresentou a maior AUC-ROC, com valor de 0,7710, enquanto a divisão 75/25 apresentou a maior acurácia, com valor de 0,7105. No entanto, as diferenças entre os cenários são muito pequenas, o que indica que o desempenho do modelo não depende de uma divisão específica dos dados.

O recall ficou próximo de 0,70 nos três experimentos. Isso significa que o modelo conseguiu identificar cerca de 70% dos pacientes com doença cardiovascular. No entanto, ainda há ocorrência de falsos negativos, ou seja, pacientes com doença cardiovascular classificados como saudáveis. Esse ponto é relevante no contexto clínico, pois falsos negativos podem representar risco maior do que falsos positivos.

A matriz de confusão reforça essa análise. No experimento 80/20, por exemplo, o modelo classificou corretamente 4.738 pacientes com doença, mas deixou de identificar 2.058 pacientes doentes. Esse resultado indica que o modelo possui capacidade preditiva relevante, mas ainda pode ser aprimorado, especialmente com técnicas de ajuste de limiar, otimização de hiperparâmetros ou comparação com outros algoritmos.

A validação cruzada apresentou média de AUC-ROC de 0,7713 e baixo desvio padrão, indicando estabilidade do modelo. Assim, conclui-se que o Random Forest é uma escolha adequada como primeiro modelo para o problema, oferecendo desempenho consistente, interpretabilidade parcial por importância das variáveis e robustez diante das características da base.

# Pipeline de pesquisa e análise de dados

O pipeline de pesquisa e análise de dados foi estruturado da seguinte forma:

1. Definição do problema:
   - classificação binária para prever a presença ou ausência de doença cardiovascular.

2. Carregamento da base:
   - utilização da base `cardio_train_sem_valores_invalidos.csv`, previamente tratada para remoção de valores clinicamente inválidos.

3. Preparação dos dados:
   - remoção da coluna `id`;
   - conversão da idade de dias para anos;
   - criação da variável IMC;
   - verificação de valores clinicamente inválidos;
   - verificação de valores nulos;
   - análise do balanceamento da variável alvo.

4. Transformação dos dados:
   - padronização das variáveis contínuas com `StandardScaler`;
   - manutenção das variáveis categóricas e binárias já codificadas numericamente.

5. Separação dos dados:
   - realização de treino/teste: 80/20;
   - uso de estratificação para preservar a proporção da variável alvo.

6. Treinamento:
   - construção de pipeline com `ColumnTransformer` e `RandomForestClassifier`;
   - treinamento do modelo.

7. Avaliação:
   - cálculo de AUC-ROC, acurácia, precisão, recall e F1-score;
   - geração da matriz de confusão;
   - geração da curva ROC;
   - análise da importância das variáveis;
   - validação cruzada estratificada com 5 folds.

8. Registro dos resultados:
   - salvamento dos gráficos;
   - salvamento das métricas em arquivos CSV;
   - consolidação das métricas em `metricas_random_forest_treino_teste.csv`.

Esse pipeline permite que o experimento seja reproduzido, comparado e expandido em etapas futuras, incluindo novos modelos, ajuste de hiperparâmetros e análise mais aprofundada dos falsos negativos.

## Observações importantes

Todas as tarefas realizadas nesta etapa deverão ser registradas em formato de texto junto com suas explicações de forma a apresentar os códigos desenvolvidos e também, o código deverá ser incluído, na íntegra, na pasta "src".
