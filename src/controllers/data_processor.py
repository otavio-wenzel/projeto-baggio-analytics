import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.schemas import formatar_rotulo


def configurar_estilo_grafico(tema: str = "Claro"):
    plt.rcdefaults()
    params_base = {
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.dpi": 115
    }
    if tema == "Escuro":
        plt.style.use("dark_background")
        plt.rcParams.update({
            **params_base,
            "axes.facecolor": "#1e293b",
            "figure.facecolor": "#0f172a",
            "axes.edgecolor": "#475569",
            "grid.color": "#334155",
            "text.color": "#f8fafc",
            "axes.labelcolor": "#f8fafc",
            "xtick.color": "#e2e8f0",
            "ytick.color": "#e2e8f0",
            "legend.facecolor": "#1e293b",
            "legend.edgecolor": "#475569"
        })
    else:
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            **params_base,
            "axes.facecolor": "#ffffff",
            "figure.facecolor": "#ffffff",
            "axes.edgecolor": "#cbd5e1",
            "grid.color": "#e2e8f0",
            "text.color": "#0f172a",
            "axes.labelcolor": "#0f172a",
            "xtick.color": "#1e293b",
            "ytick.color": "#1e293b",
            "legend.facecolor": "#f8fafc",
            "legend.edgecolor": "#cbd5e1"
        })


def formatar_eixo_x(ax, labels, largura_max: int = 16, rotacao: int = 25):
    textos_limpos = [textwrap.fill(formatar_rotulo(lbl), width=largura_max) for lbl in labels]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        textos_limpos,
        rotation=rotacao,
        ha="right",
        rotation_mode="anchor",
        fontsize=8.5
    )


def detectar_picos_automaticos(serie_mensal: pd.Series, nomes_meses: list, sensibilidade_std: float = 0.5):
    valores = serie_mensal.values.astype(float)
    if len(valores) == 0 or valores.max() == 0:
        return 0.0, 0.0, []

    media = float(np.mean(valores))
    desvio = float(np.std(valores))
    if desvio == 0:
        return media, media, []

    limiar = media + (sensibilidade_std * desvio)
    indices_pico = [i for i, val in enumerate(valores) if val > limiar]
    if not indices_pico:
        return media, limiar, []

    intervalos = []
    ini = indices_pico[0]
    fim = indices_pico[0]
    for idx in indices_pico[1:]:
        if idx == fim + 1:
            fim = idx
        else:
            intervalos.append((ini, fim))
            ini = idx
            fim = idx
    intervalos.append((ini, fim))

    picos_formatados = []
    for numero, (i_ini, i_fim) in enumerate(intervalos, start=1):
        periodo_str = nomes_meses[i_ini] if i_ini == i_fim else f"{nomes_meses[i_ini]}–{nomes_meses[i_fim]}"
        picos_formatados.append((i_ini, i_fim, f"{numero}º Pico Detectado ({periodo_str})"))

    return media, limiar, picos_formatados


def preparar_base_cruzada(
    df_imoveis: pd.DataFrame,
    df_contratos: pd.DataFrame,
    multiplicador_iqr: float = 1.5
):
    if df_imoveis.empty or df_contratos.empty:
        return pd.DataFrame(), pd.DataFrame(), 0.0

    df_merged = pd.merge(
        df_contratos,
        df_imoveis,
        left_on="Referência Imóvel",
        right_on="Referência",
        how="inner",
        suffixes=("_contrato", "_imovel")
    )

    for col_dt in ["Data de Início", "Data de Rescisão", "Data de Término"]:
        df_merged[col_dt] = pd.to_datetime(df_merged[col_dt], errors="coerce")

    df_merged["Meses_Locados"] = (df_merged["Data de Rescisão"] - df_merged["Data de Início"]).dt.days / 30.0
    df_merged["Quebra_Antecipada"] = np.where(
        df_merged["Data de Rescisão"] < df_merged["Data de Término"], 1, 0
    )

    col_status = "Status_imovel" if "Status_imovel" in df_merged.columns else "Status"
    df_merged["Proprietario_Evadiu"] = np.where(df_merged[col_status] == "Baixado", 1, 0)

    df_valid = df_merged[df_merged["Meses_Locados"] > 0].copy()
    if df_valid.empty:
        return df_valid, df_valid, 0.0

    q1 = df_valid["Meses_Locados"].quantile(0.25)
    q3 = df_valid["Meses_Locados"].quantile(0.75)
    iqr = q3 - q1
    limite_superior = float(q3 + multiplicador_iqr * iqr)

    df_limpo = df_valid[df_valid["Meses_Locados"] <= limite_superior].copy()
    return df_valid, df_limpo, limite_superior