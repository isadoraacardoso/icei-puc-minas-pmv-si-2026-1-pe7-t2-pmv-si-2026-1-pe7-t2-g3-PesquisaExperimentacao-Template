# =============================================================================
# LIMPEZA DE DADOS — DATASET CARDIO TRAIN
# Projeto: Predição de Doença Cardiovascular
# Objetivo: gerar uma base limpa e reprodutível para a EDA removendo apenas
# valores clinicamente inválidos, sem remover outliers pelo método IQR.
# =============================================================================

from pathlib import Path
import pandas as pd


# =============================================================================
# 1. CONFIGURAÇÃO DE CAMINHOS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

ARQUIVO_ENTRADA = BASE_DIR / "cardio_train.csv"
ARQUIVO_SAIDA = BASE_DIR / "cardio_train_sem_valores_invalidos.csv"
ARQUIVO_RELATORIO = BASE_DIR / "cardio_train_relatorio_limpeza.csv"
ARQUIVO_REMOVIDOS = BASE_DIR / "cardio_train_linhas_removidas.csv"


# =============================================================================
# 2. CARREGAMENTO DOS DADOS
# =============================================================================

print("=" * 70)
print("LIMPEZA DE DADOS — CARDIO TRAIN")
print("=" * 70)

if not ARQUIVO_ENTRADA.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {ARQUIVO_ENTRADA}\n"
        "Verifique se o arquivo cardio_train.csv está na mesma pasta deste script."
    )

df = pd.read_csv(ARQUIVO_ENTRADA, sep=";")

print(f"\nRegistros iniciais: {len(df):,}")
print(f"Colunas iniciais: {df.shape[1]}")


# =============================================================================
# 3. FUNÇÃO AUXILIAR PARA APLICAR FILTROS
# =============================================================================

relatorio = []
linhas_removidas = []


def aplicar_filtro(df_atual, criterio, descricao):
    """
    Aplica um filtro booleano ao DataFrame, registra quantos registros foram
    removidos e armazena as linhas removidas para análise posterior.
    """
    antes = len(df_atual)

    removidos_df = df_atual[~criterio].copy()
    removidos_df["criterio_remocao"] = descricao

    if len(removidos_df) > 0:
        linhas_removidas.append(removidos_df)

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
# - peso plausível.
#
# Nesta versão, os outliers pelo método IQR NÃO são removidos.
# O CSV cardio_train_linhas_removidas.csv conterá apenas as linhas removidas
# pelos critérios clinicamente inválidos abaixo.
# =============================================================================

df_limpo = df.copy()

# Criar variáveis derivadas para análise e relatório.
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
# 4.2 OUTLIERS PELO MÉTODO IQR
# =============================================================================
# Decisão metodológica:
# Os outliers pelo método IQR NÃO são removidos nesta versão.
# Motivo: em um problema de doença cardiovascular, valores extremos de pressão,
# peso e IMC podem representar pacientes reais e clinicamente relevantes.
# =============================================================================

print("\n[INFO] Outliers pelo método IQR não serão removidos nesta versão.")
print("       A base final terá apenas valores clinicamente inválidos removidos.")


# =============================================================================
# 5. SALVAMENTO DOS ARQUIVOS
# =============================================================================

df_limpo.to_csv(ARQUIVO_SAIDA, sep=";", index=False)

df_relatorio = pd.DataFrame(relatorio)
df_relatorio.to_csv(ARQUIVO_RELATORIO, sep=";", index=False)

if linhas_removidas:
    df_removidos = pd.concat(linhas_removidas, ignore_index=True)
    df_removidos.to_csv(ARQUIVO_REMOVIDOS, sep=";", index=False)
else:
    df_removidos = pd.DataFrame()
    df_removidos.to_csv(ARQUIVO_REMOVIDOS, sep=";", index=False)

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

print(f"\nLinhas removidas salvas em:")
print(f"  {ARQUIVO_REMOVIDOS}")
