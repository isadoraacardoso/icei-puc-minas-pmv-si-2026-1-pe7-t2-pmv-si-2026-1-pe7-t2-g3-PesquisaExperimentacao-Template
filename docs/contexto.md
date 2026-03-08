# Introdução

As doenças cardiovasculares representam uma das principais causas de mortalidade no mundo. De acordo com a Organização Mundial da Saúde (OMS), milhões de pessoas morrem anualmente em decorrência dessas doenças, muitas vezes devido ao diagnóstico tardio ou à dificuldade em identificar fatores de risco de forma precoce. Nesse contexto, a utilização de técnicas de análise de dados e aprendizado de máquina é uma abordagem promissora para auxiliar profissionais da saúde na identificação de padrões associados à presença de doenças cardíacas.

A priori, o projeto de pesquisa e experimentação tem como objetivo investigar e experimentar modelos de aprendizado de máquina aplicados ao dataset Heart Disease, disponível no repositório da UCI Machine Learning Repository. Esse conjunto de dados reúne informações clínicas e demográficas de pacientes coletadas em diferentes instituições hospitalares, incluindo características como idade, sexo, pressão arterial, níveis de colesterol e resultados de exames cardíacos.

Além disso, temos como proposta explorar esses dados para compreender quais variáveis apresentam maior relação com a ocorrência de doença cardíaca e avaliar modelos capazes de prever a presença da doença a partir dessas características. A investigação busca contribuir para a compreensão de como técnicas de mineração de dados podem apoiar processos de análise em contextos médicos.

Assim, nosso projeto se insere no contexto de experimentação acadêmica em ciência de dados e aprendizado de máquina, com foco na análise de dados clínicos e na avaliação de modelos preditivos que possam auxiliar na identificação de riscos associados a doenças cardiovasculares.

## Problema

As doenças cardíacas são frequentemente diagnosticadas por meio da análise conjunta de diversos exames clínicos e históricos médicos dos pacientes. No entanto, a interpretação dessas informações pode ser complexa, especialmente quando diferentes variáveis precisam ser consideradas simultaneamente para identificar padrões associados à presença da doença.

Em muitos casos, médicos e profissionais de saúde precisam analisar um grande volume de informações clínicas, como idade, níveis de colesterol, pressão arterial, frequência cardíaca e resultados de exames específicos. A dificuldade em identificar rapidamente relações entre essas variáveis pode tornar o processo de diagnóstico mais demorado ou menos preciso.

Nesse contexto, técnicas de análise de dados e aprendizado de máquina podem auxiliar na identificação de padrões que indicam maior probabilidade de ocorrência de doença cardíaca. A partir da análise de dados históricos de pacientes, é possível explorar modelos capazes de reconhecer combinações de características associadas ao diagnóstico da doença.

O presente projeto utiliza o dataset Heart Disease como base para experimentação. Esse conjunto de dados contém registros clínicos de pacientes e inclui diversas variáveis médicas que podem estar relacionadas à presença de doença cardíaca. O problema investigado neste trabalho está relacionado à dificuldade de identificar, de forma sistemática, quais fatores apresentam maior influência no diagnóstico e se modelos de aprendizado de máquina são capazes de realizar essa previsão de forma confiável.

A investigação será conduzida em um contexto acadêmico, utilizando ferramentas de análise de dados e bibliotecas de aprendizado de máquina amplamente utilizadas na área de ciência de dados. O objetivo é explorar o potencial dessas técnicas para apoiar a análise de dados médicos e compreender melhor os padrões presentes no conjunto de dados estudado.

## Questão de pesquisa

A questão de pesquisa orienta o desenvolvimento deste projeto e define o foco principal da investigação.

Diante do problema apresentado, busca-se investigar se é possível utilizar técnicas de aprendizado de máquina para identificar padrões relevantes nos dados clínicos de pacientes e prever a presença de doença cardíaca com base nessas informações.

Dessa forma, a questão de pesquisa que orienta este trabalho é:

É possível utilizar modelos de aprendizado de máquina para prever a presença de doença cardíaca em pacientes com base em características clínicas presentes no dataset Heart Disease?

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

O dataset Heart Disease, amplamente utilizado em estudos acadêmicos de ciência de dados e aprendizado de máquina, oferece um conjunto de informações clínicas que permite explorar a relação entre diferentes características dos pacientes e a presença de doença cardíaca. A análise desse conjunto de dados possibilita investigar como modelos computacionais podem aprender padrões presentes nos dados e contribuir para a previsão de diagnósticos.

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

<p><b>1.Heart‑Disease‑Prediction(abhinavsaurabh)</b></p>
<p><b>Problema e contexto:</b> Um projeto de análise preditiva com o objetivo de identificar a probabilidade de doenças cardíacas com base em dados de pacientes, utilizando técnicas de aprendizado de máquina</p>
 
<b>Dataset:</b> UCI Heart Disease Dataset.

⦁	Origem: UCI Machine Learning Repository

⦁	Tamanho: 303 registros (subconjunto Cleveland)

⦁	Período: estudos clínicos analisados em pesquisas recentes até 2024

<b>Variáveis:</b>

⦁	idade

⦁	sexo

⦁	tipo de dor no peito

⦁	pressão arterial

⦁	colesterol

⦁	glicose em jejum

⦁	frequência cardíaca máxima

⦁	angina induzida por exercício

⦁	depressão ST

⦁	entre outras (14 atributos)
<p><b>Abordagem:</b> Implementação de múltiplos algoritmos de aprendizado de máquina, incluindo Regressão Logística, Random Forest e Máquinas de Vetores de Suporte (SVM).</p>

<p><b>Métricas:</b> Comparação de desempenho usando acurácia, precisão, recall e pontuação ROC-AUC</p>

<b>Resultados:</b>

⦁	Melhor modelo: O Random Forest alcançou uma precisão de 92%, superando os demais modelos.

⦁	Principais fatores preditivos: Idade, nível de colesterol e frequência cardíaca máxima foram identificados como fatores preditivos significativos.

<p><b>2. Predição de Doença Cardíaca com Random Forest, SVM e Regressão Logística</b></p>

<b>Problema e contexto:</b> Desenvolver modelos de ML para prever a presença de doença cardíaca usando dados clínicos de pacientes.

<b>Dataset:</b> UCI Heart Disease Dataset

⦁	Origem: UCI Machine Learning Repository

⦁	Tamanho: 303 registros (subconjunto Cleveland)

⦁	Período: estudos clínicos analisados em pesquisas recentes até 2024

⦁	Variáveis:

⦁	idade

⦁	sexo

⦁	tipo de dor no peito

⦁	pressão arterial

⦁	colesterol

⦁	glicose em jejum

⦁	frequência cardíaca máxima

⦁	angina induzida por exercício

⦁	depressão ST

⦁	entre outras (14 atributos)

<b>Abordagem:</b>

⦁	Random Forest

⦁	Support Vector Machine (SVM)

⦁	Regressão Logística

<b>Métricas:</b> 

⦁	Accuracy

⦁	Precision

⦁	Recall

⦁	F1-score

<b>Resultados:</b>

⦁	Random Forest: 89,7% accuracy (melhor modelo)

⦁	SVM: 87,0%

⦁	Regressão Logística: 84,2%

<p><b>3. Framework de Machine Learning para Predição de Doença Cardíaca</b></p>
 
<b>Problema e contexto:</b> Criar um sistema de apoio ao diagnóstico clínico para prever doenças cardíacas.

<b>Dataset:</b> Heart Disease.

⦁	Origem: UCI Machine Learning Repository

⦁	Tamanho: 303 pacientes

⦁	Variáveis: 13 variáveis clínicas + variável alvo

⦁	Atributos principais: idade, colesterol, pressão arterial, tipo de dor torácica, frequência cardíaca máxima, etc.

⦁	Período: estudos clínicos analisados em pesquisas recentes até 2025.

<b>Abordagem:</b> 

⦁	Random Forest

⦁	KNN

⦁	Regressão Logística

⦁	Otimização com GridSearchCV e RandomizedSearchCV

<b>Métricas:</b>

⦁	Accuracy

⦁	Precision

⦁	Recall

⦁	F1-score

⦁	Confusion matrix

<b>Resultados:</b> 

⦁	Random Forest: 91% accuracy

⦁	F1-score: 0,89

⦁	Limitação: tamanho reduzido do dataset e baixa generalização.

<p><b>4. Ensemble Framework para Predição de Doenças Cardiovasculares</b></p>
   
<b>Problema e contexto:</b> Desenvolver um modelo ensemble para melhorar a precisão na predição de doenças cardíacas.

<b>Dataset:</b> 

⦁	Combinação de datasets médicos: Cleveland, Hungary, Switzerland, Statlog

⦁	Variáveis: frequência cardíaca máxima, colesterol, dor no peito, glicemia etc.

⦁	Período: estudos clínicos analisados em pesquisas recentes até 2025.

<b>Abordagem:</b> 

⦁	Stacking ensemble

⦁	Random Forest

⦁	XGBoost

⦁	Extra Trees

<b>Métricas:</b> 

⦁	Accuracy

⦁	ROC-AUC

⦁	F1-score

⦁	Sensitivity e Specificity

<b>Resultados:</b> 

⦁	Accuracy: 92,34%

⦁	Melhor desempenho que modelos individuais.

<p><b>5. Predição de Doença Cardíaca com Dataset Integrado</b></p>
   
<b>Problema e contexto:</b> Prever doenças cardíacas combinando dados de múltiplos hospitais.

<b>Dataset:</b> 

⦁	Cleveland

⦁	Hungary

⦁	Switzerland

⦁	Long Beach VA

⦁	~11 variáveis clínicas

⦁	Período: estudos clínicos analisados em pesquisas recentes até 2024.

<b>Abordagem:</b>

⦁	Random Forest

⦁	SVM

⦁	Logistic Regression

<b>Métricas:</b> 

⦁	Accuracy

⦁	Precision

⦁	Recall

<b>Resultados:</b> 

⦁	Random Forest: 92,9% accuracy

⦁	SVM: 89,7%

⦁	Logistic Regression: 86,1%


<b>Síntese Crítica</b>

<p>Após a análise dos projetos, observou-se que eles buscam prever doenças cardiovasculares por meio da utilização de técnicas de aprendizagem de máquina, possibilitando a avaliação do risco dessas doenças em tempo real. Em geral, os estudos apresentam abordagens semelhantes, utilizando algoritmos de aprendizagem supervisionada, como Random Forest, Regressão Logística e SVM, além de métricas de avaliação comuns, como accuracy, precision, recall e F1-score. Entretanto, diferem quanto aos dados utilizados, já que alguns trabalhos utilizam o subconjunto Cleveland, enquanto outros empregam diferentes conjuntos de dados, como Hungary, Statlog ou Long Beach VA.</p>

<p>Em relação às lacunas identificadas, destaca-se que muitos estudos trabalham com conjuntos de dados relativamente pequenos, o que pode limitar a generalização dos resultados. Além disso, raramente são discutidas questões éticas e possíveis vieses, como desigualdades relacionadas a sexo, idade ou características populacionais, aspectos importantes para garantir maior confiabilidade e equidade nos modelos desenvolvidos.</p>

<p>O projeto em desenvolvimento também tem como objetivo a previsão de doenças cardíacas, porém pretende explorar o potencial do Cardiovascular Disease Dataset (Cardio Train) para validar modelos em cenários hospitalares e acadêmicos, contribuindo para a detecção preventiva de doenças cardiovasculares e ampliando as possibilidades de aplicação dessas soluções na área da saúde.</p>

### Ferramentas inteligentes permitidas
Você pode utilizar: Perplexity, SciSpace, Elicit, Research Rabbit, Litmaps.
Use-as para descoberta, organização e triagem de literatura. 

**Atenção:** 
* Sempre acesse a fonte original (PDF/artigo) antes de citar; verifique números e conclusões.
* Registre DOI/URL oficial e dados bibliográficos completos.
* Evite “alucinações” das ferramentas: desconfie de referências sem DOI ou que você não consiga localizar oficialmente.
* Use as ferramentas inteligentes para mapear redes de citação (Research Rabbit), mapas de tópicos (Litmaps), filtrar por período e gerar resumos iniciais (Perplexity/SciSpace/Elicit).
* Leia os trabalhos mais promissores e descarte estudos fora de escopo.

> **Links Úteis**:
> - [Google Scholar](https://scholar.google.com/)
> - [IEEE Xplore](https://ieeexplore.ieee.org/Xplore/home.jsp)
> - [Science Direct](https://www.sciencedirect.com/)
> - [ACM Digital Library](https://dl.acm.org/)

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

<img width="3780" height="2670" alt="canvas_analitico_doenca_cardiovascular_70k" src="https://github.com/user-attachments/assets/184c16d1-dbfa-4f44-a707-c8333fd57e38" />

# Vídeo de apresentação da Etapa 01

Nesta etapa, o grupo deverá produzir um vídeo de 5 a 8 minutos apresentando o trabalho realizado, no qual cada integrante deve dizer seu nome e apresentar uma parte do conteúdo desenvolvido, garantindo que todos participem ativamente da gravação. A ausência de participação de qualquer membro resultará em penalização na nota final desta etapa. Recomenda-se que o grupo elabore previamente um roteiro para organizar a ordem das falas, distribuir o tempo de forma equilibrada e assegurar que todos os tópicos relevantes sejam apresentados de maneira clara e objetiva.

# Referências

Inclua todas as referências (livros, artigos, sites, etc) utilizados no desenvolvimento do trabalho utilizando o padrão ABNT.

> **Links Úteis**:
> - [Padrão ABNT PUC Minas](https://portal.pucminas.br/biblioteca/index_padrao.php?pagina=5886)
> - [Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset?resource=download)
> - [Organização Mundial da Saúde](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))
