# Implantação da solução

Nesta etapa, deverá ser realizada e descrita em detalhes a implantação da solução em ambiente de computação em nuvem, incluindo o planejamento da capacidade operacional com base em modelagem matemática e simulação do sistema, a escolha e configuração do provedor de nuvem, o empacotamento e publicação da aplicação, bem como a execução de testes que comprovem seu correto funcionamento.

Importante!! É fundamental que a aplicação em produção (_deploy_ em nuvem) esteja preparada preparada para realizar a inferência dinâmica, utilizando os modelos previamente treinados, com interface para entrada de novos dados fornecidos pelo usuário, de modo que as **predições sejam realizadas em tempo de execução**, sem reprocessamento ou atualização do treinamento.


# Apresentação da solução

Nesta seção, deve ser produzido um vídeo de até 15 minutos apresentando o escopo geral do projeto, um resumo das etapas desenvolvidas, a demonstração da solução publicada e as conclusões finais, destacando aprendizados, impacto e possibilidades de melhorias.

# É IMPRESCINDÍVEL: 
* Atualizar o arquivo **CITATION.cff** disponível no diretório raiz do repositório
* Atualizar as **Instruções de utilização** no arquivo read.me

# Relatório de Implantação e Infraestrutura em Nuvem - Eixo 7 (Grupo 3)

Este documento apresenta o planejamento, a modelagem matemática de capacidade, a escolha da infraestrutura e o processo de deploy da solução de inferência preditiva desenvolvida para o projeto de Pesquisa e Experimentação em Sistemas de Informação.

---

## 1. Avaliação de Provedores de Serviço em Nuvem e Justificativa de Escolha

Para a definição da infraestrutura de hospedagem da solução de inferência preditiva, realizou-se uma análise comparativa entre os três principais provedores de computação em nuvem do mercado (*Hyperscalers*), considerando critérios de viabilidade financeira (camada gratuita), complexidade de gerência e suporte a ambientes virtuais isolados.

| Critério de Análise | Amazon Web Services (AWS) | Google Cloud Platform (GCP) | Microsoft Azure |
| :--- | :--- | :--- | :--- |
| **Serviço de Computação** | Elastic Compute Cloud (EC2) | Google Compute Engine (GCE) | Azure Virtual Machines |
| **Elegibilidade Free Tier** | 750 horas/mês gratuitas por 12 meses (instâncias `t2.micro` / `t3.micro`). | 1 instância `e2-micro` gratuita por mês por tempo ilimitado (regiões específicas). | 750 horas/mês gratuitas por 12 meses (instâncias `B1s`). |
| **Configuração de Rede** | Grupos de Segurança (*Security Groups*) baseados em regras Stateful. | Regras de Firewall VPC baseadas em Tags e IPs de origem. | Network Security Groups (NSG) baseados em regras de prioridade. |
| **Facilidade de Integração** | Elevada através da AWS CLI e gerenciamento de chaves PEM. | Elevada via Google Cloud CLI integrado à conta Google. | Elevada através de extensões nativas do VS Code. |

### Justificativa da Escolha da AWS (EC2)
A escolha da **Amazon Web Services (AWS)** sustentou-se no modelo de Infraestrutura como Serviço (IaaS) por meio do **AWS EC2**. Embora o GCP apresente uma excelente interface gerenciada, a AWS oferece o ambiente computacional purista ideal para projetos de experimentação em Sistemas de Informação. A utilização de instâncias da família `t3.micro` permitiu o provisionamento de um servidor virtual Linux Ubuntu inteiramente custeado pelo programa *AWS Free Tier*, eliminando custos operacionais para o grupo. Ademais, o ecossistema de segurança da AWS, controlado via *Security Groups*, viabilizou o isolamento estrito de portas de rede, garantindo que apenas o tráfego essencial do ecossistema Python chegue à aplicação.

---

## 2. Planejamento de Capacidade Operacional e Modelagem Matemática

O dimensionamento do servidor de produção utilizou a **Teoria das Filas (Modelagem de Kendall M/M/1)** para simular o comportamento do sistema sob carga computacional e prever possíveis gargalos estruturais antes da implantação física na AWS.

### 2.1. Premissas de Carga do Sistema
O modelo de classificação (baseado no algoritmo *Random Forest*) foi treinado sobre um conjunto de dados de aproximadamente 70.000 registros. Após o processo de otimização e poda de hiperparâmetros, o arquivo binário final serializado gerou um impacto leve em armazenamento em disco. Contudo, o sistema deve ser dimensionado para responder a requisições concorrentes em tempo de execução.

Definem-se os seguintes parâmetros para a modelagem:
* **Capacidade de Atendimento do Servidor ($\mu$):** Testes locais de estresse demonstraram que o tempo médio de processamento de uma inferência dinâmica (leitura do payload JSON, conversão em vetor numérico, execução do método `predict` e retorno HTTP) é de **$150\text{ms}$** ($0,15\text{ segundos}$). Portanto, a taxa de serviço de uma única vCPU operando em Thread isolada é:
    $$\mu = \frac{1}{0,15} \approx 6,67 \text{ requisições por segundo (RPS)}$$
    $$\mu_{\text{minuto}} = 6,67 \times 60 = 400 \text{ requisições por minuto (RPM)}$$

* **Taxa de Chegada Estimada ($\lambda$):** Estima-se um cenário de pico acadêmico com 15 usuários simultâneos operando a interface, onde cada usuário submete novos dados de formulário a cada 10 segundos ($6\text{ requisições/minuto}$ por usuário).
    $$\lambda_{\text{total}} = 15 \times 6 = 90 \text{ requisições por minuto (RPM)}$$
    $$\lambda = \frac{90}{60} = 1,5 \text{ requisições por segundo (RPS)}$$

### 2.2. Aplicação do Modelo Matemático
Para que o sistema opere em regime de estabilidade e não entre em colapso computacional, a taxa de utilização do servidor ($\rho$) deve ser estritamente menor que 1 ($\rho < 1$).

* **Fator de Utilização da CPU ($\rho$):**
    $$\rho = \frac{\lambda}{\mu} = \frac{1,5}{6,67} \approx 0,2249 \implies 22,5\% \text{ de ocupação da CPU}$$

* **Número Médio de Requisições no Sistema ($L$):**
    $$L = \frac{\rho}{1 - \rho} = \frac{0,2249}{1 - 0,2249} \approx 0,29 \text{ requisições}$$

* **Tempo Médio Total de Resposta Esperado pelo Usuário ($W$):**
    $$W = \frac{1}{\mu - \lambda} = \frac{1}{6,67 - 1,5} = \frac{1}{5,17} \approx 0,193 \text{ segundos (193 ms)}$$

### 2.3. Determinação do Gargalo Operacional
O limite teórico máximo de saturação da instância configurada ocorre quando $\lambda \to \mu$. Matematicamente, a instância de 1 vCPU atingirá seu gargalo operacional estrito ao processar **400 requisições por minuto**. A partir desse ponto crítico, o tempo de espera na fila ($W$) tende ao infinito, gerando estouro de memória (*Out of Memory*) e indisponibilidade do serviço. Como a demanda projetada é de 90 RPM, a folga operacional da instância `t2.micro` é de **$77,5\%$**, validando a escolha do hardware sem custos.

---

## 3. Implantação do Artefato em Produção (Inferência Dinâmica)

A arquitetura de software implementada em produção foi estruturada para garantir o desacoplamento entre a fase de treinamento do modelo e a fase de consumo operacional.

### 3.1. Salvamento e Persistência do Modelo
O modelo preditivo *Random Forest*, após atingir as métricas de validação satisfatórias em ambiente de desenvolvimento, foi persistido e exportado utilizando a biblioteca de serialização de alta performance `joblib`. Este método preserva a estrutura de grafos e os pesos matemáticos das árvores de decisão em um arquivo binário estruturado:

```python
import joblib
# Persistência do modelo e do pipeline de transformação dentro da pasta src
joblib.dump(modelo_final, 'src/modelo_rf.pkl')
