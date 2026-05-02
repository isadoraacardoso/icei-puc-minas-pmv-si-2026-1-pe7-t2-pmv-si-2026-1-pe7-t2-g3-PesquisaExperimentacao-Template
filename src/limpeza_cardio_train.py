# =============================================================================
# LIMPEZA DE DADOS — DATASET CARDIO TRAIN
# Projeto: Predição de Doença Cardiovascular
# Objetivo: gerar uma base limpa e reprodutível para a EDA sem outliers
# =============================================================================

from pathlib import Path
import pandas as pd


# =============================================================================
# 1. CONFIGURAÇÃO DE CAMINHOS
# =============================================================================

# Considerando que o csv está está dentro da pasta src/
#BASE_DIR = Path(__file__).resolve().parents[1]

#ARQUIVO_ENTRADA = BASE_DIR / "cardio_train.csv"
#ARQUIVO_SAIDA = Path(__file__).resolve().parent / "cardio_train_sem_outliers.csv" # Salva na mesma pasta do script
#ARQUIVO_RELATORIO = Path(__file__).resolve().parent / "cardio_train_relatorio_limpeza.csv"     # Salva na mesma pasta do script

BASE_DIR = Path(__file__).resolve().parent

ARQUIVO_ENTRADA = BASE_DIR / "cardio_train.csv"
ARQUIVO_SAIDA = BASE_DIR / "cardio_train_sem_outliers.csv"
ARQUIVO_RELATORIO = BASE_DIR / "cardio_train_relatorio_limpeza.csv"


# =============================================================================
# 2. CARREGAMENTO DOS DADOS
# =============================================================================

print("=" * 70)
print("LIMPEZA DE DADOS — CARDIO TRAIN")
print("=" * 70)

if not ARQUIVO_ENTRADA.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {ARQUIVO_ENTRADA}\n"
        "Verifique se o arquivo cardio_train.csv está na raiz do repositório."
    )

df = pd.read_csv(ARQUIVO_ENTRADA, sep=";")

print(f"\nRegistros iniciais: {len(df):,}")
print(f"Colunas iniciais: {df.shape[1]}")


# =============================================================================
# 3. FUNÇÃO AUXILIAR PARA APLICAR FILTROS
# =============================================================================

relatorio = []

def aplicar_filtro(df_atual, criterio, descricao):
    """
    Aplica um filtro booleano ao DataFrame e registra quantos registros foram removidos.
    """
    antes = len(df_atual)
    df_filtrado = df_atual[criterio].copy()
    depois = len(df_filtrado)
    removidos = antes - depois

    relatorio.append({
        "criterio": descricao,
        "registros_antes": antes,
        "registros_removidos": removidos,
        "registros_restantes": depois,
        "percentual_removido": round((removidos / antes) * 100, 4) if antes > 0 else 0
    })

    print(f"\nCritério: {descricao}")
    print(f"  Antes     : {antes:,}")
    print(f"  Removidos : {removidos:,}")
    print(f"  Restantes : {depois:,}")

    return df_filtrado


# =============================================================================
# 4. CRITÉRIOS DE LIMPEZA
# =============================================================================
# Critérios baseados nas inconsistências identificadas na eda-etapa2.py:
# - pressão sistólica válida;
# - pressão diastólica válida;
# - sistólica maior ou igual à diastólica;
# - altura plausível;
# - peso plausível etc...
#
# REMOÇÃO SEQUENCIAL: cada critério é aplicado sobre o DataFrame resultante do critério anterior.
# =============================================================================

df_limpo = df.copy()

# Criar variáveis derivadas para remoção de outliers
df_limpo["age_years"] = (df_limpo["age"] / 365.25).round(1)
df_limpo["bmi"] = (df_limpo["weight"] / ((df_limpo["height"] / 100) ** 2)).round(2)


# =============================================================================
# 4.1 REMOÇÃO DE VALORES CLINICAMENTE INVÁLIDOS
# =============================================================================

df_limpo = aplicar_filtro(
    df_limpo,
    (df_limpo["ap_hi"] > 0) & (df_limpo["ap_hi"] <= 300),
    "Remover pressão sistólica inválida: ap_hi <= 0 ou ap_hi > 300"
)

df_limpo = aplicar_filtro(
    df_limpo,
    (df_limpo["ap_lo"] > 0) & (df_limpo["ap_lo"] <= 200),
    "Remover pressão diastólica inválida: ap_lo <= 0 ou ap_lo > 200"
)

df_limpo = aplicar_filtro(
    df_limpo,
    df_limpo["ap_hi"] >= df_limpo["ap_lo"],
    "Remover registros em que pressão sistólica é menor que a diastólica"
)

df_limpo = aplicar_filtro(
    df_limpo,
    (df_limpo["height"] >= 100) & (df_limpo["height"] <= 220),
    "Remover altura fora do intervalo plausível: height < 100 ou height > 220"
)

df_limpo = aplicar_filtro(
    df_limpo,
    (df_limpo["weight"] >= 30) & (df_limpo["weight"] <= 200),
    "Remover peso fora do intervalo plausível: weight < 30 ou weight > 200"
)


# =============================================================================
# 4.2 REMOÇÃO DE OUTLIERS PELO MÉTODO IQR
# =============================================================================

def remover_outliers_iqr(df_atual, coluna, nome_variavel):
    """
    Remove outliers de uma coluna usando o método IQR.
    Mantém apenas valores entre:
    Q1 - 1.5 * IQR e Q3 + 1.5 * IQR.
    """
    Q1 = df_atual[coluna].quantile(0.25)
    Q3 = df_atual[coluna].quantile(0.75)
    IQR = Q3 - Q1

    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    descricao = (
        f"Remover outliers de {nome_variavel} pelo método IQR "
        f"[{limite_inferior:.2f}, {limite_superior:.2f}]"
    )

    return aplicar_filtro(
        df_atual,
        (df_atual[coluna] >= limite_inferior) & (df_atual[coluna] <= limite_superior),
        descricao
    )


df_limpo = remover_outliers_iqr(df_limpo, "age_years", "Idade em anos")
df_limpo = remover_outliers_iqr(df_limpo, "height", "Altura")
df_limpo = remover_outliers_iqr(df_limpo, "weight", "Peso")
df_limpo = remover_outliers_iqr(df_limpo, "ap_hi", "Pressão Sistólica")
df_limpo = remover_outliers_iqr(df_limpo, "ap_lo", "Pressão Diastólica")
df_limpo = remover_outliers_iqr(df_limpo, "bmi", "IMC")


# =============================================================================
# 5. SALVAMENTO DOS ARQUIVOS
# =============================================================================

df_limpo.to_csv(ARQUIVO_SAIDA, sep=";", index=False)

df_relatorio = pd.DataFrame(relatorio)
df_relatorio.to_csv(ARQUIVO_RELATORIO, sep=";", index=False)

print("\n" + "=" * 70)
print("RESUMO FINAL DA LIMPEZA")
print("=" * 70)

print(f"Registros iniciais : {len(df):,}")
print(f"Registros finais   : {len(df_limpo):,}")
print(f"Total removido     : {len(df) - len(df_limpo):,}")
print(f"Percentual removido: {((len(df) - len(df_limpo)) / len(df)) * 100:.2f}%")

print(f"\nBase limpa salva em:")
print(f"  {ARQUIVO_SAIDA}")

print(f"\nRelatório da limpeza salvo em:")
print(f"  {ARQUIVO_RELATORIO}")

# Alerta para conferir compatibilidade com o relatório textual
#REGISTROS_ESPERADOS_RELATORIO = 66488

#if len(df_limpo) != REGISTROS_ESPERADOS_RELATORIO:
#    print("\nATENÇÃO:")
#    print(
#        f"O relatório menciona {REGISTROS_ESPERADOS_RELATORIO:,} registros finais, "
#        f"mas este script gerou {len(df_limpo):,}."
#    )
#    print(
#        "Isso indica que os critérios usados no relatório precisam ser ajustados "
#        "ou que o número informado no texto deve ser atualizado."
#    )
#else:
#    print("\nConferência OK:")
#    print("O número final de registros coincide com o relatório.")