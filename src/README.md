# src — Pipelines de Predição de Doença Cardiovascular

## Dependências

```bash
pip install -r requirements.txt
```

## Dataset

Todos os scripts esperam o arquivo `cardio_train_sem_valores_invalidos.csv` na pasta `src/`.  
Execute a partir da **raiz do repositório**:

```bash
python src/<numero>_<script>.py
```

---

## Ordem de execução

| # | Script | Descrição | Entrada | Saída principal |
|---|--------|-----------|---------|-----------------|
| 1 | `1_eda-etapa2.py` | Análise exploratória inicial | `cardio_train.csv` | Gráficos em `graficos/` |
| 2 | `2_limpeza_cardio_sem_remover_iqr.py` | Limpeza de registros inválidos | `cardio_train.csv` | `cardio_train_sem_valores_invalidos.csv` |
| 3 | `3_eda-etapa2-sem-outliers.py` | EDA pós-limpeza | `cardio_train_sem_valores_invalidos.csv` | Gráficos em `graficos/` |
| 4 | `4_pipeline_etapa3.py` | Pipeline Random Forest | `cardio_train_sem_valores_invalidos.csv` | Gráficos, `modelo_cardio.onnx` |
| 5 | `5_pipeline_svm.py` | Pipeline SVM (6 configurações) | `cardio_train_sem_valores_invalidos.csv` | Gráficos, `graficos/resultados_svm.csv` |
| 6 | `6_pipeline_logistic_regression.py` | Pipeline Regressão Logística | `cardio_train_sem_valores_invalidos.csv` | Gráficos com prefixo `lr_` |

---

## Pré-processamento compartilhado (scripts 4, 5 e 6)

Todos os três modelos seguem o mesmo pipeline lógico:

1. Remoção da coluna `id`
2. Conversão de `age` (dias) → `age_years` (anos)
3. Feature engineering: `IMC = weight / (height/100)²`
4. Remoção de registros clinicamente inválidos (pressão, altura, peso)
5. Divisão estratificada 80/20 (`stratify=y`, `random_state=42`)
6. `StandardScaler` aplicado apenas nas colunas contínuas via `ColumnTransformer`

Colunas contínuas: `age_years`, `height`, `weight`, `ap_hi`, `ap_lo`, `IMC`  
Colunas categóricas (passthrough): `gender`, `cholesterol`, `gluc`, `smoke`, `alco`, `active`

---

## Artefatos gerados

```
graficos/
├── 11_curva_roc.png              # ROC — Random Forest
├── 12_matriz_confusao.png        # Matriz — Random Forest
├── 13_feature_importance.png     # Importância features — Random Forest
├── lr_curva_roc.png              # ROC — Regressão Logística
├── lr_matriz_confusao.png        # Matriz — Regressão Logística
├── lr_coeficientes.png           # Coeficientes — Regressão Logística
├── matriz_SVM_*.png              # Matrizes por configuração SVM
├── comparacao_svm_recall.png     # Ranking SVM por Recall
├── comparacao_svm_acuracia.png   # Ranking SVM por Acurácia
├── comparacao_svm_auc.png        # Ranking SVM por AUC-ROC
└── resultados_svm.csv            # Tabela completa de métricas SVM
modelo_cardio.onnx                # Modelo Random Forest exportado
```
