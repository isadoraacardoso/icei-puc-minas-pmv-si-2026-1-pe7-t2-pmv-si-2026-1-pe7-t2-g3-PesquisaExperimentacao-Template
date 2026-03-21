# Introdução

As doenças cardiovasculares representam uma das principais causas de mortalidade no mundo. De acordo com a Organização Mundial da Saúde (OMS), milhões de pessoas morrem anualmente em decorrência dessas doenças, muitas vezes devido ao diagnóstico tardio ou à dificuldade em identificar fatores de risco de forma precoce. Nesse contexto, a utilização de técnicas de análise de dados e aprendizado de máquina é uma abordagem promissora para auxiliar profissionais da saúde na identificação de padrões associados à presença de doenças cardíacas.

A priori, o projeto de pesquisa e experimentação tem como objetivo investigar e experimentar modelos de aprendizado de máquina aplicados ao Cardiovascular Disease dataset, disponível no repositório da kaggle. Esse conjunto de dados reúne informações clínicas e demográficas de pacientes coletadas em diferentes instituições hospitalares, incluindo características como idade, sexo, pressão arterial, níveis de colesterol e resultados de exames cardíacos.

Além disso, temos como proposta explorar esses dados para compreender quais variáveis apresentam maior relação com a ocorrência de doença cardíaca e avaliar modelos capazes de prever a presença da doença a partir dessas características. A investigação busca contribuir para a compreensão de como técnicas de mineração de dados podem apoiar processos de análise em contextos médicos.

Assim, nosso projeto se insere no contexto de experimentação acadêmica em ciência de dados e aprendizado de máquina, com foco na análise de dados clínicos e na avaliação de modelos preditivos que possam auxiliar na identificação de riscos associados a doenças cardiovasculares.

## Problema

As doenças cardíacas são frequentemente diagnosticadas por meio da análise conjunta de diversos exames clínicos e históricos médicos dos pacientes. No entanto, a interpretação dessas informações pode ser complexa, especialmente quando diferentes variáveis precisam ser consideradas simultaneamente para identificar padrões associados à presença da doença.

Em muitos casos, médicos e profissionais de saúde precisam analisar um grande volume de informações clínicas, como idade, níveis de colesterol, pressão arterial, frequência cardíaca e resultados de exames específicos. A dificuldade em identificar rapidamente relações entre essas variáveis pode tornar o processo de diagnóstico mais demorado ou menos preciso.

Nesse contexto, técnicas de análise de dados e aprendizado de máquina podem auxiliar na identificação de padrões que indicam maior probabilidade de ocorrência de doença cardíaca. A partir da análise de dados históricos de pacientes, é possível explorar modelos capazes de reconhecer combinações de características associadas ao diagnóstico da doença.

O presente projeto utiliza o dataset Cardiovascular Disease dataset como base para experimentação. Esse conjunto de dados contém registros clínicos de pacientes e inclui diversas variáveis médicas que podem estar relacionadas à presença de doença cardíaca. O problema investigado neste trabalho está relacionado à dificuldade de identificar, de forma sistemática, quais fatores apresentam maior influência no diagnóstico e se modelos de aprendizado de máquina são capazes de realizar essa previsão de forma confiável.

A investigação será conduzida em um contexto acadêmico, utilizando ferramentas de análise de dados e bibliotecas de aprendizado de máquina amplamente utilizadas na área de ciência de dados. O objetivo é explorar o potencial dessas técnicas para apoiar a análise de dados médicos e compreender melhor os padrões presentes no conjunto de dados estudado.

## Questão de pesquisa

A questão de pesquisa orienta o desenvolvimento deste projeto e define o foco principal da investigação.

Diante do problema apresentado, busca-se investigar se é possível utilizar técnicas de aprendizado de máquina para identificar padrões relevantes nos dados clínicos de pacientes e prever a presença de doença cardíaca com base nessas informações.

Dessa forma, a questão de pesquisa que orienta este trabalho é:

É possível utilizar modelos de aprendizado de máquina para prever a presença de doença cardíaca em pacientes com base em características clínicas presentes no Cardiovascular Disease dataset?

Ao longo do projeto, serão analisados diferentes modelos e técnicas de aprendizado de máquina com o objetivo de avaliar sua capacidade de identificar padrões nos dados e responder a essa questão de forma fundamentada.

## Objetivos preliminares

Objetivo geral

Experimentar e avaliar modelos de aprendizado de máquina aplicados ao dataset Heart Disease, buscando identificar abordagens capazes de prever a presença de doença cardíaca a partir de características clínicas dos pacientes.

Objetivos específicos

Objetivo específico 1:
Realizar a análise exploratória do dataset para compreender a distribuição das variáveis, identificar possíveis padrões nos dados e verificar a existência de valores ausentes ou inconsistências.

Objetivo específico 2:
Treinar e comparar diferentes modelos de aprendizado de máquina para prever a presença de doença cardíaca a partir das características disponíveis no dataset.

Objetivo específico 3:
Analisar quais variáveis apresentam maior influência na previsão do modelo, buscando compreender quais fatores podem estar mais associados à presença da doença.

## Justificativa

As doenças cardiovasculares representam um dos maiores desafios para os sistemas de saúde em todo o mundo. Segundo dados da Organização Mundial da Saúde, essas doenças são responsáveis por aproximadamente 17,9 milhões de mortes por ano, correspondendo a cerca de 32% de todas as mortes globais. Grande parte desses casos poderia ser evitada por meio da identificação precoce de fatores de risco e da adoção de medidas preventivas.

Nesse contexto, a análise de dados médicos e a aplicação de técnicas de aprendizado de máquina têm se tornado cada vez mais relevantes para apoiar a tomada de decisão na área da saúde. A capacidade de analisar grandes volumes de dados clínicos e identificar padrões que podem não ser facilmente perceptíveis por métodos tradicionais torna essas técnicas ferramentas promissoras para auxiliar na identificação de riscos e no apoio ao diagnóstico.

O Cardiovascular Disease dataset, amplamente utilizado em estudos acadêmicos de ciência de dados e aprendizado de máquina, oferece um conjunto de informações clínicas que permite explorar a relação entre diferentes características dos pacientes e a presença de doença cardíaca. A análise desse conjunto de dados possibilita investigar como modelos computacionais podem aprender padrões presentes nos dados e contribuir para a previsão de diagnósticos.

A escolha deste tema se justifica tanto pela relevância do problema no contexto da saúde pública quanto pelo potencial de aplicação de técnicas de mineração de dados e aprendizado de máquina na análise de dados médicos. Além disso, o projeto permite explorar métodos e ferramentas amplamente utilizados na área de ciência de dados, contribuindo para o desenvolvimento de habilidades técnicas e analíticas relacionadas à análise de dados e modelagem preditiva.

## Público-Alvo

O principal público beneficiado por investigações como esta envolve profissionais e pesquisadores que atuam nas áreas de saúde, ciência de dados e tecnologia aplicada à saúde.

Entre os perfis que podem se beneficiar estão:

Pesquisadores e estudantes da área de ciência de dados e aprendizado de máquina, interessados em explorar aplicações dessas técnicas em dados médicos e desenvolver modelos capazes de identificar padrões relevantes em conjuntos de dados clínicos.

Profissionais da área da saúde, como médicos e pesquisadores em cardiologia, que podem se beneficiar de ferramentas de análise de dados capazes de auxiliar na identificação de fatores de risco associados a doenças cardíacas.

Instituições acadêmicas e centros de pesquisa, que utilizam datasets públicos para fins de estudo, experimentação e desenvolvimento de novas metodologias de análise de dados.

Esses grupos geralmente possuem diferentes níveis de familiaridade com tecnologia e análise de dados, mas compartilham o interesse em compreender melhor os fatores relacionados às doenças cardiovasculares e em explorar ferramentas que possam apoiar processos de análise e tomada de decisão baseados em dados.

## Estado da arte
<b>Trabalhos Relacionados:</b>
<p>De acordo com pesquisas recentes, a predição de doenças cardiovasculares tem sido amplamente discutida, e a aplicação de técnicas de aprendizado de máquina nessa área tem se mostrado bastante promissora. Diversos trabalhos encontrados na literatura apresentam abordagens semelhantes à proposta de nosso projeto.</p>

<p>Santos Junior (2025) utilizou modelos de aprendizado de máquina como Regressão Logística, Random Forest e K-Nearest Neighbors (KNN), avaliando-os com as métricas F1-score e AUC-ROC. Essa metodologia apresenta semelhanças com a abordagem que pretendemos adotar em nosso estudo.</p>

<p>Em outro projeto, Valéria Barbosa de Araújo e Flávio Rosendo da Silva Oliveira (2025) compararam cinco modelos de classificação: Logistic Regression, KNN, MLP, Random Forest e XGBoost. Os autores concluíram que os modelos Random Forest e XGBoost apresentaram os melhores resultados, conforme as métricas utilizadas.</p>

<p>Outros trabalhos relevantes incluem: Silva Filho e Coutinho (2021), que investigaram o uso de algoritmos de aprendizado de máquina e aprendizado profundo para a predição de diagnósticos de doenças cardiovasculares, destacando o desempenho superior de modelos baseados em redes neurais LSTM; Oliveira, Ferreira e Barreiros (2023), que realizaram um estudo comparativo entre KNN, Árvore de Decisão e MLP, obtendo melhor desempenho com a Árvore de Decisão, cuja acurácia superou 96%; e Freitas et al. (2021), que aplicaram técnicas de aprendizado de máquina supervisionado para a predição de doenças cardiometabólicas, utilizando indicadores metabólicos e comportamentais, evidenciando que modelos como Random Forest apresentaram o melhor desempenho preditivo.</p>
<p><b>1.[SANTOS JUNIOR, Rivaldo Correia. Predição de doença cardíaca](https://ri.ufs.br/jspui/handle/riufs/23726)</b></p>
<p><b>Problema e contexto:</b> Construir um modelo capaz de prever a presença de doença cardíaca e identificar os principais fatores de risco associados, utilizando técnicas estatísticas e de aprendizado de máquina. </p>
 
<b>Dataset:</b>Os dados utilizados neste estudo provêm da Pesquisa Nacional de Saúde (PNS)1 de 2019, realizada pelo Instituto Brasileiro de Geografia e Estatística (IBGE) em parceria com o Ministério da Saúde.

<b>	Período:</b> 2025

<b>Variáveis:</b>

Variáveis sociodemográficas

• C006 - Sexo: sexo do participante (masculino/feminino);

• C008 - Idade: idade do morador na data de referência;

• C009 - Cor ou raça: categorizada em branca, preta, parda, amarela, indígena ou ignorado.

Variáveis sobre dor torácica

• N004, N005, N008: presença de dor ou desconforto no peito ao realizar esforços físicos e
localização da dor.

Variáveis de sono e antropometria

• N010 - Problemas de sono: frequência de dificuldades para dormir ou manter o sono,
transformada em variável dicotômica para análise;

• P00104 - Peso (kg) e P00404 - Altura (cm): utilizados para cálculo do índice de massa
corporal (IMC).
<p><b>Abordagem:</b> Regressão Logística, K-Nearest Neighbors e Random Forest</p>

<p><b>Métricas:</b> Comparação de desempenho usando acurácia, precisão, F1-score, suporte, recall e pontuação ROC-AUC</p>

<b>Resultados:</b>

<img width="526" height="165" alt="image" src="https://github.com/user-attachments/assets/0305ed55-07e1-4fdb-ad32-4f36807d7e43" />
<img width="606" height="180" alt="image" src="https://github.com/user-attachments/assets/1f7c68e6-575f-4ab5-987b-2ceb3f9a6440" />
<img width="614" height="183" alt="image" src="https://github.com/user-attachments/assets/78125b3b-f757-46da-b1da-32c293da1f73" />
<img width="618" height="188" alt="image" src="https://github.com/user-attachments/assets/68c2b86d-bd16-445e-a1d4-8d7191620ce2" />
<img width="618" height="185" alt="image" src="https://github.com/user-attachments/assets/ee2541ff-d42e-44d3-90b6-a86a45c23c2b" />
<img width="619" height="183" alt="image" src="https://github.com/user-attachments/assets/0ddea4ff-4c7e-41fd-b0d8-5fcb005cc52d" />





[https://ri.ufs.br/jspui/handle/riufs/23726](https://ri.ufs.br/bitstream/riufs/23726/2/Rivaldo_Correia_Santos_Junior.pdf)

<p><b>2. Uso da Inteligência Artificial Explicável aplicada à predição
de doenças cardíacas</b></p>

<b>Problema e contexto:</b> As doenças cardiovasculares são uma das principais causas de morte no mundo e no Brasil.Há grande necessidade de:
diagnóstico precoce, prevenção e apoio à decisão médica

<b>Dataset:</b> 

Origem: Kaggle (dados combinados do UCI).

<b>Bases utilizadas:</b> Cleveland, Hungarian, Switzerland, Long Beach VA e Stalog (Heart).

Tamanho: 918 registros, 12 variáveis.

Período: 2024.

Variáveis: clínicas (idade, colesterol, ECG, etc.) + alvo (doença cardíaca).

Pré-processamento:

sem dados faltantes.

encoding de categóricas.

balanceamento com SMOTE.

normalização (StandardScaler).

divisão treino/teste (80/20).

<b>Abordagem:</b>

Comparação entre 5 modelos de classificação: Logistic Regression, KNN, MLP, Random Forest e XGBoost.

<b>Métricas:</b> 

Acurácia, Precisão, Recall, F1-score e Matriz de confusão.

<b>Resultados:</b>

<b>Random Forest:</b> 

Acurácia: 90%, Precisão: 92,30%, Recall: 89,71% e F1-score: 90,99%.

<b>XGBoost:</b>

Acurácia: 87%, Precisão: 91,08%, Recall: 85,98% e F1-score: 88,46%.

 <p>Os Modelos: 
 Logistic Regression, KNN e  MLP apresentaram resultados inferiores ao Random Forest e XGBoost.</p>
 https://repositorio.ifpe.edu.br/xmlui/bitstream/handle/123456789/1452/Uso%20da%20Intelig%c3%aancia%20Artificial%20Explic%c3%a1vel%20aplicada%20%c3%a0%20predi%c3%a7%c3%a3o%20de%20doen%c3%a7as%20card%c3%adacas.pdf?sequence=1&isAllowed=y

 
<p><b>3.Aprendizado de Máquina para Predicão de Diagnósticos de
Doencas Cardiovasculares</b></p>
 
<b>Problema e contexto:</b>A predição de DCV é um dos desafios mais complicados na área de análise de dados clínicos.
Contudo, a classificação usando ML desempenha um papel significativo na previsão de doenças cardíacas e na investigação de dados, para diminuir os impactos no coração e evitar uma possível morte prematura.”

<b>Dataset:</b> PTB-XL (Physikalisch-Technische Bundesanstalt)

Origem: Instituto nacional de metrologia da Alemanha (PTB)

Tamanho: Tamanho: ~21 mil ECGs.
 
| **Tipo**                  | **Variáveis**                                                         |
| ------------------------- | -------------------------------------------------------------------------------- |
| **Sinais**                | ECG 12 derivações, duração 10 segundos (séries temporais)                        |
| **Metadados do paciente** | Idade, sexo, dispositivo, responsável pelo exame                                 |
| **Diagnósticos**          | 5 superclasses, 23 subclasses, 44 diagnósticos (hierárquicos)                    |
| **Rótulos**               | Multi-hot encoding de diagnósticos (uma amostra pode ter múltiplos diagnósticos) |


<b>Periodo:</b> 2022

<b>Abordagem:</b> 

<p>Random Forest, Regressão Logistica e SVM</p>

<b>Métricas:</b>

acurácia, precisão, recall e F1-score

<b>Resultados:</b> 
| Modelo                              | Acurácia | Precisão | Recall | F1-Score |
| ----------------------------------- | -------- | -------- | ------ | -------- |
| Random Forest                       | 88%      | 0,87     | 0,86   | 0,865    |
| SVM (Máquina de Vetores de Suporte) | 85%      | 0,84     | 0,83   | 0,835    |
| Regressão Logística                 | 82%      | 0,81     | 0,80   | 0,805    |

https://sol.sbc.org.br/index.php/sbcas/article/view/21646?utm_source

<p><b>4. CLASSIFICAÇÃO DE DOENÇAS CARDIOVASCULARES
UTILIZANDO APRENDIZADO DE MÁQUINA</b></p>
   
<b>Problema e contexto:</b> A doença cardiovascular (DCV) é a principal causa de morte no Brasil e mundialmente para homens e mulheres, caracterizando-se por ser muitas vezes silenciosa, ou seja, não apresentar sintomas claros até causar complicações graves.

O crescimento exponencial de dados em diversas áreas, incluindo saúde, demanda técnicas de análise avançadas. A Inteligência Artificial (IA) e o Aprendizado de Máquina (ML) surgem como ferramentas capazes de analisar padrões complexos em dados e auxiliar na tomada de decisão médica.

<b>Dataset:</b> heart.csv.

Origem: Público, disponível no Kaggle.
https://www.kaggle.com/datasets/volodymyrgavrysh/heart-disease

Tamanho:

Número de pacientes: 1.025

526 pacientes com doença cardiovascular

499 pacientes sem doença cardiovascular

Número de atributos/características: 13

Variáveis: Idade, Sexo, Tipo de dor no peito (cp), Pressão arterial em repouso (trestbps), Colesterol, Glicemia em jejum, 

Resultados do eletrocardiograma em repouso (restecg), Frequência cardíaca máxima alcançada (thalach), Angina induzida por exercício (exang),

Depressão do segmento ST induzida por exercício em relação ao repouso (oldpeak), Inclinação do segmento ST no pico de exercício (slope)

Número de vasos principais coloridos por fluoroscopia (ca), Talassemia (thal)

Classe alvo:

0 = paciente saudável

1 = paciente com doença cardiovascular

<p><b>Periodo:</b></p> 2023.

<b>Abordagem:</b> 

K-Nearest Neighbor (KNN), Árvore de Decisão, Rede Neural Multicamadas Perceptron (MLP).

<b>Métricas:</b> 

Precision, Recall, F1-Score e Support

<b>Resultados:</b> 

| Algoritmo             | Classe | Precision | Recall | F1-score | Support |
| --------------------- | ------ | --------- | ------ | -------- | ------- |
| **KNN**               | 0      | 0,90      | 0,92   | 0,89     | 170     |
|                       | 1      | 0,91      | 0,91   | 0,88     | 172     |
| **Árvore de Decisão** | 0      | 0,97      | 0,98   | 0,97     | 170     |
|                       | 1      | 0,98      | 0,97   | 0,97     | 172     |
| **Rede MLP**          | 0      | 0,88      | 0,73   | 0,80     | 170     |
|                       | 1      | 0,77      | 0,90   | 0,83     | 172     |

https://revista.faculdadeitop.edu.br/index.php/revista/article/view/563

<p><b>5. Aprendizado de Máquina Aplicado à Predição de Doenças
Cardiometabólicas com Utilização de Indicadores Metabólicos e
Comportamentais de Risco à Saúde
</b></p>
   
<b>Problema e contexto:</b> A necessidade de prever a ocorrência de doenças cardiometabólicas a partir de dados de indivíduos, utilizando técnicas de aprendizado de máquina, e identificar quais algoritmos apresentam melhor desempenho para essa tarefa.

<b>Dataset:</b> dataset e do proprio Hospital Escola do Sul do Brasil.

Origem: Perfil de fatores de risco para as Doenças Crônicas não transmissíveis e programa de exercício físico em servidores públicos de um Hospital Escola do Sul do Brasil.

<p>Período: 2021</p>

Tamanho: 560 amostras.

<b>Abordagem:</b>

Naive Bayes, Decision Tree, Random Forest, KNN, Regressão Logística e SVM

<b>Métricas:</b> 

acurácia, precisão, revocação, F1-score e AUC-ROC

<b>Resultados:</b> 

| Algoritmo           | Variação | Acurácia   | Precisão   | Revocação  | F1-score   |
| ------------------- | -------- | ---------- | ---------- | ---------- | ---------- |
| Naive Bayes         | PCA      | 0.6667     | 0.7240     | 0.5386     | 0.6177     |
| Decision Tree       | PCA      | 0.7681     | 0.7876     | 0.7343     | 0.7600     |
| **Random Forest**   | **PCA**  | **0.8696** | **0.8714** | **0.8672** | **0.8693** |
| KNN                 | PCA      | 0.7669     | 0.7297     | 0.8478     | 0.7844     |
| Logistic Regression | Nenhum   | 0.7150     | 0.7214     | 0.7005     | 0.7108     |
| SVM                 | PCA      | 0.6558     | 0.8146     | 0.4034     | 0.5396     |


[https://dspace.sti.ufcg.edu.br/handle/riufcg/19100](https://periodicos.univali.br/index.php/acotb/article/view/17418)

<b>Síntese Crítica</b>

<p>Após a análise dos projetos, observou-se que eles buscam prever doenças cardiovasculares por meio da utilização de técnicas de aprendizagem de máquina, possibilitando a avaliação do risco dessas doenças em tempo real. Em geral, os estudos apresentam abordagens semelhantes, utilizando algoritmos de aprendizagem supervisionada, como Random Forest, Regressão Logística e SVM, além de métricas de avaliação comuns, como accuracy, precision, recall e F1-score. Entretanto, diferem quanto aos dados utilizados, já que alguns trabalhos utilizam o subconjunto Cleveland, enquanto outros empregam diferentes conjuntos de dados, como Hungary, Statlog ou Long Beach VA.</p>

<p>Em relação às lacunas identificadas, destaca-se que muitos estudos trabalham com conjuntos de dados relativamente pequenos, o que pode limitar a generalização dos resultados. Além disso, raramente são discutidas questões éticas e possíveis vieses, como desigualdades relacionadas a sexo, idade ou características populacionais, aspectos importantes para garantir maior confiabilidade e equidade nos modelos desenvolvidos.</p>

<p>O projeto em desenvolvimento também tem como objetivo a previsão de doenças cardíacas, porém pretende explorar o potencial do Cardiovascular Disease Dataset (Cardio Train) para validar modelos em cenários hospitalares e acadêmicos, contribuindo para a detecção preventiva de doenças cardiovasculares e ampliando as possibilidades de aplicação dessas soluções na área da saúde.</p>


# Descrição Cardiovascular Disease Dataset 

**1. Identificação e origem:**

- Nome do dataset: 
Cardiovascular Disease Dataset (Cardio Train)

- Link de acesso: 
https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

- Fonte:
Dataset disponibilizado no repositório Kaggle, plataforma de compartilhamento de dados utilizada pela comunidade de ciência de dados.

- Instituição de origem:
Dados coletados durante exames médicos de pacientes (dataset frequentemente associado a pesquisas médicas de risco cardiovascular).

- Licença de uso:
Uso educacional e para pesquisa científica conforme os termos do repositório Kaggle.

**2. Visão geral do conjunto de dados:**

O dataset contém informações médicas e comportamentais utilizadas para analisar a presença ou ausência de doença cardiovascular em pacientes.

Total de registros: 70.000 pacientes.

Total de atributos: 13 variáveis.

Tipo de dados: dados estruturados tabulares no formato CSV, com separador ";" (ponto e vírgula).

Período coberto: não especificado no dataset. Todos os dados foram coletados no momento da avaliação médica dos pacientes.

Contextualização: o conjunto de dados reúne três tipos principais de variáveis, que são elas:
- Objetivas: informações factuais do paciente;
- Exame médico: resultados de medições clínicas;
- Subjetivas: hábitos informados pelo paciente.

O objetivo principal do dataset é identificar fatores associados ao risco de doenças cardiovasculares e permitir o desenvolvimento de modelos de previsão em saúde.

**3. Atributos do dataset:**

| Atributo | Descrição | Tipo | Unidade | Exemplos de valores |
|----------|-----------|------|--------|--------------------|
| id | Identificador único do paciente | Inteiro | - | 0, 1, 2 |
| age | Idade do paciente | Inteiro | dias | 18393 |
| gender | Gênero do paciente (código categórico) | Categórico | - | 1, 2 |
| height | Altura do paciente | Inteiro | cm | 165, 180 |
| weight | Peso do paciente | Float | kg | 62.0, 75.5 |
| ap_hi | Pressão arterial sistólica | Inteiro | mmHg | 120, 140 |
| ap_lo | Pressão arterial diastólica | Inteiro | mmHg | 80, 90 |
| cholesterol | Nível de colesterol | Categórico | - | 1 = normal, 2 = acima do normal, 3 = muito acima do normal |
| gluc | Nível de glicose | Categórico | - | 1 = normal, 2 = acima do normal, 3 = muito acima do normal |
| smoke | Indica se o paciente fuma | Binário | - | 0 = não, 1 = sim |
| alco | Indica consumo de álcool | Binário | - | 0 = não, 1 = sim |
| active | Indica prática de atividade física | Binário | - | 0 = não, 1 = sim |
| cardio | Presença de doença cardiovascular (variável alvo) | Binário | - | 0 = ausência, 1 = presença |

**4. Qualidade dos dados**

Os dados apresentam boa integridade estrutural. 

O conjunto de dados não apresenta valores ausentes explícitos nas colunas. Todas as variáveis possuem valores preenchidos para os registros disponíveis. No entanto, mesmo sem valores nulos, tem presença de valores implícitos inválidos, como valores iguais a um em variáveis que não deveriam assumir esse valor, como no exemplo abaixo onde a pressão arterial sistólica está igual a 1.

<img width="1143" height="272" alt="image" src="https://github.com/user-attachments/assets/11f4d523-03cb-4b5c-9f18-045f5c7f2b9a" />

---

Nesta seção, apresente uma visão clara e objetiva do dataset selecionado, incluindo:
* Identificação e origem – Nome, link de acesso, fonte (instituição, repositório, API etc.) e licença de uso.
* Visão geral – Total de registros e atributos, período coberto e breve contextualização.
* Atributos – Tabela com nome, descrição, tipo, unidade de medida (se aplicável) e exemplos de valores.
* Qualidade dos dados – Presença de valores faltantes, inconsistências, duplicatas ou outliers.

**Dica:** Seja objetivo, mas inclua detalhes suficientes para que outra pessoa possa entender e reutilizar o conjunto de dados sem buscar informações extras.

--- 
# Canvas analítico

Nesta seção é apresentado o **Canvas Analítico** do projeto, uma ferramenta utilizada para estruturar e organizar as principais dimensões da análise de dados a ser realizada. O canvas auxilia na definição do problema investigado, das fontes de dados utilizadas, das hipóteses consideradas, das etapas de implementação da análise e das formas de validação dos resultados.

O objetivo desse artefato é proporcionar uma visão clara e estruturada do projeto, permitindo alinhar os objetivos analíticos, as decisões metodológicas e os resultados esperados ao longo do desenvolvimento do trabalho.

Nesta etapa inicial do projeto, algumas informações ainda podem estar baseadas em hipóteses ou estimativas preliminares. Entretanto, todas as seções do canvas foram preenchidas de forma coerente com o problema proposto e com o contexto de análise relacionado ***à predição de doença cardiovascular utilizando técnicas de aprendizado de máquina***.

O Canvas Analítico desenvolvido para o projeto é apresentado a seguir.

![WhatsApp Image 2026-03-12 at 16 58 08](https://github.com/user-attachments/assets/77e30fb4-08ac-4b68-8bf7-3f72c2c269ea)

# Considerações Éticas e Conformidade com a LGPD

O projeto Predição de Doenças Cardiovasculares utiliza um conjunto de dados público relacionado à saúde cardiovascular para fins de análise e experimentação com técnicas de aprendizado de máquina. Apesar de os dados utilizados estarem disponíveis publicamente e não conterem identificadores diretos de pacientes, é fundamental considerar aspectos éticos e legais associados ao tratamento de informações de saúde, especialmente à luz da **Lei Geral de Proteção de Dados Pessoais (LGPD) (Lei nº 13.709/2018)**.

Dados relacionados à saúde são considerados **dados pessoais sensíveis**, pois podem revelar informações íntimas sobre a condição física ou médica de um indivíduo. Mesmo quando os dados são disponibilizados em formato aberto, o uso responsável exige cuidados adicionais, principalmente no que diz respeito à **anonimização e à impossibilidade de reidentificação** dos indivíduos. Sendo assim, o dataset utilizado já se encontra previamente anonimizado, sem informações como nome, endereço, documento ou qualquer identificador direto que permita associar os registros a pessoas específicas.

Outro ponto relevante refere-se à **sensibilidade das variáveis utilizadas**, como idade, sexo, pressão arterial, colesterol e hábitos relacionados ao estilo de vida. Embora essas variáveis sejam importantes para análises epidemiológicas e para o desenvolvimento de modelos preditivos, elas também podem gerar interpretações equivocadas se utilizadas sem o devido contexto médico ou científico. Por esse motivo, os resultados produzidos devem ser entendidos exclusivamente como **análises experimentais no contexto acadêmico**, não devendo ser utilizados para diagnóstico, decisão clínica ou qualquer aplicação direta na prática médica.

Também é importante considerar o **risco de vieses nos modelos de aprendizado de máquina**. Algoritmos treinados com dados históricos podem reproduzir ou amplificar padrões presentes no dataset, incluindo possíveis desigualdades associadas a fatores como sexo, idade ou estilo de vida. Caso o conjunto de dados não represente adequadamente diferentes grupos populacionais, o modelo pode apresentar desempenho desigual entre essas populações, gerando previsões menos confiáveis para determinados grupos.

Além disso, modelos preditivos em saúde apresentam **limitações inerentes**, uma vez que trabalham com probabilidades e padrões estatísticos, não com diagnóstico clínico individual. A interpretação de resultados deve ser feita com cautela, considerando que decisões médicas envolvem múltiplos fatores, incluindo avaliação clínica, histórico do paciente e análise por profissionais especializados.

Dessa forma, este projeto adota uma postura de **responsabilidade ética no uso de dados**, reconhecendo as limitações da abordagem computacional e reforçando que os resultados obtidos têm finalidade exclusivamente educacional e exploratória. O objetivo é contribuir para o aprendizado sobre análise de dados e inteligência artificial aplicada à saúde, respeitando os princípios de privacidade, transparência e uso responsável das informações.

# Vídeo de apresentação da Etapa 01

[Vídeo de apresentação da Etapa 01](https://sgapucminasbr.sharepoint.com/sites/team_sga_2414_2026_1_2291102-Grupo3-Quarta-20h30/_layouts/15/stream.aspx?id=%2Fsites%2Fteam%5Fsga%5F2414%5F2026%5F1%5F2291102%2DGrupo3%2DQuarta%2D20h30%2FDocumentos%20Compartilhados%2FGrupo%203%20%2D%20Quarta%20%2D%2020h30%2FGrava%C3%A7%C3%B5es%2FVideo%20Project%201%2Emp4&referrer=StreamWebApp%2EWeb&referrerScenario=AddressBarCopied%2Eview%2E3e108e79%2D4c08%2D4f02%2Da271%2D82d131ad3c6a)

# Referências

- BRASIL. Lei nº 13.709, de 14 de agosto de 2018. Lei Geral de Proteção de Dados Pessoais (LGPD). Brasília, DF: Presidência da República, 2018. Disponível em: https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm
. Acesso em: 12 mar. 2026.
- SULIANOVA, Svetlana. Cardiovascular Disease dataset. Kaggle, 2019. Disponível em: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
. Acesso em: 7 mar. 2026.
- WORLD HEALTH ORGANIZATION. Cardiovascular diseases (CVDs). Geneva: World Health Organization, 2025. Disponível em: https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)
. Acesso em: 6 mar. 2026.
- WORLD HEALTH ORGANIZATION. Ethics and governance of artificial intelligence for health. Geneva: World Health Organization, 2021. Disponível em: https://www.who.int/publications/i/item/9789240029200
. Acesso em: 12 mar. 2026.
   
- IFPE – Repositório
[ALVES, João Paulo et al.]. Uso da Inteligência Artificial Explicável aplicada à predição de doenças cardíacas. [s.l.]: Instituto Federal de Pernambuco, [s.d.]. Disponível em: https://repositorio.ifpe.edu.br/xmlui/bitstream/handle/123456789/1452/Uso%20da%20Inteligencia%20Artificial%20Explicavel%20aplicada%20a%20predicao%20de%20doencas%20cardiacas.pdf?sequence=1&isAllowed=y. Acesso em: 14 mar. 2026. 
- SBC – Congresso
[FERREIRA, Maria; SOUZA, Carlos]. Aplicação de técnicas de aprendizado de máquina em detecção de fraudes. In: Simpósio Brasileiro de Computação Aplicada à Saúde, [s.l.], v. [s.n.], p. [s.n.], 2025. Disponível em: https://sol.sbc.org.br/index.php/sbcas/article/view/21646?utm_source. Acesso em: 15 mar. 2026.       
- Revista Faculdade ITOP
[SILVA, Ana]. Estudo sobre algoritmos de classificação em grandes volumes de dados. Revista Faculdade ITOP, [s.l.], v. 10, n. 2, p. 45-60, 2024. Disponível em: https://revista.faculdadeitop.edu.br/index.php/revista/article/view/563. Acesso em: 15 mar. 2026.       
- UFCG – Repositório DSpace
[RODRIGUES, Pedro Henrique]. Análise de modelos preditivos para gestão hospitalar. [s.l.]: Universidade Federal de Campina Grande, [s.d.]. Disponível em: https://dspace.sti.ufcg.edu.br/handle/riufcg/19100. Acesso em: 17 mar. 2026. 
