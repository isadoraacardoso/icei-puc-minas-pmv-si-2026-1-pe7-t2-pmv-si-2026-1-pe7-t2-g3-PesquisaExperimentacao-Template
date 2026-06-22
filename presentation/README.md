# Apresentação da Solução

Nesta seção, é apresentado o consolidado do projeto experimental desenvolvido pelo Grupo 3 para a disciplina de Pesquisa e Experimentação em Sistemas de Informação (Eixo 7), seguido pelo registro audiovisual de encerramento e demonstração prática.

---

## 1. Resumo Executivo do Projeto Desenvolvido

O projeto consistiu no desenvolvimento, validação e implantação de uma solução de Inteligência Artificial focada em análises preditivas de doenças cardíacas, utilizando técnicas avançadas de Ciência de Dados e Engenharia de Software. O ciclo de vida do projeto compreendeu as seguintes fases macro:

### 1.1. Engenharia de Dados e Análise Exploratória
O estudo utilizou uma base de dados robusta contendo aproximadamente **70.000 registros históricos**. Foi realizado um pipeline rigoroso de preparação de dados, envolvendo o tratamento de valores ausentes, remoção de *outliers*, normalização de variáveis numéricas e codificação (*encoding*) de variáveis categóricas. A análise exploratória permitiu identificar correlações estatísticas significativas que serviram de insumo para a seleção das melhores características (*Feature Selection*).

### 1.2. Modelagem Estocástica e Treinamento
Para resolver o problema de classificação proposto, o grupo testou diferentes arquiteturas de aprendizado de máquina, selecionando o algoritmo **Random Forest (Floresta Aleatória)** devido à sua alta capacidade de generalização e resiliência ao sobreajuste (*overfitting*). Após rodar rotinas de otimização de hiperparâmetros (*Grid Search* / *Random Search*), o modelo final atingiu métricas de acurácia, precisão e *F1-Score* altamente satisfatórias. O modelo foi então serializado em formato binário estruturado (`modelo_rf.pkl`) para persistência dos pesos matemáticos.

### 1.3. Engenharia de Software e Desenvolvimento da Interface
Com o objetivo de entregar os resultados do modelo de forma acessível ao usuário final, foi desenvolvida uma aplicação web utilizando o framework **Streamlit** em linguagem Python. A interface foi projetada com foco em usabilidade (UX/UI), coletando os parâmetros de entrada do usuário de maneira intuitiva e estruturada para acionar as predições.

### 1.4. Infraestrutura em Nuvem e Inferência Dinâmica
O deploy definitivo da solução foi consolidado na plataforma **Amazon Web Services (AWS)**, utilizando uma instância virtualizada de computação **EC2 (t2.micro)** operando em ambiente Linux Ubuntu sob as diretrizes da camada gratuita (*Free Tier*). A arquitetura foi desenhada para realizar **inferência dinâmica em tempo de execução**: o modelo de aprendizado é carregado na memória RAM do servidor uma única vez no *bootstrap* do sistema. Quando o usuário insere novos dados na interface e solicita a análise, o servidor processa a resposta em milissegundos, realizando o cálculo preditivo em tempo real sem a necessidade de reprocessar o histórico ou reatualizar o treinamento do algoritmo.

---

## 2. Vídeo de Apresentação Final da Solução

O vídeo abaixo possui duração de até 15 minutos e sintetiza os resultados da pesquisa experimental desenvolvida. 



https://github.com/user-attachments/assets/29c2bd29-60f1-4e5e-8589-3210c302d971


