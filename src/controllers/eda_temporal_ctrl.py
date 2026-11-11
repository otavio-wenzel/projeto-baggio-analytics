"""
Camada Controller: Análises Exploratórias Temporais e de Motivos (A1, A2 e A3).
"""

import textwrap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.schemas import formatar_rotulo
from src.controllers.data_processor import detectar_picos_automaticos


def plot_a1_escalada_anual(
    df_imoveis: pd.DataFrame,
    intervalo_anos: tuple = (2022, 2026),
    tipos_imovel: list = None,
    formato_grafico: str = "Barras Verticais",
    metrica: str = "Quantidade Absoluta"
):
    df = df_imoveis.copy()
    df["Data de Baixa_dt"] = pd.to_datetime(df["Data de Baixa"], errors="coerce")
    baixados = df[df["Status"] == "Baixado"].dropna(subset=["Data de Baixa_dt"]).copy()
    baixados["Ano_Baixa"] = baixados["Data de Baixa_dt"].dt.year.astype(int)

    if tipos_imovel:
        baixados = baixados[baixados["Tipo"].isin(tipos_imovel)]

    ano_ini, ano_fim = intervalo_anos
    baixados = baixados[(baixados["Ano_Baixa"] >= ano_ini) & (baixados["Ano_Baixa"] <= ano_fim)]

    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    evol = baixados["Ano_Baixa"].value_counts().sort_index()

    if evol.empty:
        ax.text(0.5, 0.5, "Sem registros no intervalo selecionado", ha="center", va="center")
        return fig

    anos_lbl = [str(int(a)) for a in evol.index]
    valores = (evol.values / evol.values.sum() * 100).round(1) if metrica == "Participação (%)" else evol.values
    sufixo = "%" if metrica == "Participação (%)" else ""

    if formato_grafico == "Pizza":
        ax.pie(evol.values, labels=anos_lbl, autopct="%1.0f%%", startangle=90, colors=sns.color_palette("Blues_r", len(evol)))
        ax.set_title(f"Proporção de Imóveis Baixados ({ano_ini} a {ano_fim})", fontweight="bold")
    elif formato_grafico == "Linha com Marcadores":
        ax.plot(anos_lbl, valores, marker="o", lw=2.5, markersize=6.5, color="#2563eb")
        for i, v in enumerate(valores):
            ax.annotate(f"{v}{sufixo}", (i, v), xytext=(0, 5), textcoords="offset points", ha="center", fontweight="bold", fontsize=8.5)
        ax.set_title(f"Curva de Evolução de Baixas ({ano_ini} a {ano_fim})", fontweight="bold")
        ax.set_xlabel("Ano da Baixa")
        ax.set_ylabel(metrica)
        ax.margins(y=0.2)
    else:
        paleta = sns.color_palette("YlOrRd", len(evol))
        bars = ax.bar(anos_lbl, valores, color=paleta, edgecolor="black", width=0.5)
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.1f}{sufixo}" if sufixo else f"{int(h)}", (b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontweight="bold", fontsize=8.5)
        ax.set_title(f"Escalada Anual de Retirada de Imóveis ({ano_ini} a {ano_fim})", fontweight="bold")
        ax.set_xlabel("Ano da Baixa")
        ax.set_ylabel(metrica)
        ax.margins(y=0.2)

    plt.tight_layout()
    return fig


def plot_a2_sazonalidade_mensal(
    df_imoveis: pd.DataFrame,
    intervalo_anos: tuple = (2022, 2026),
    intervalo_meses: tuple = (1, 12),
    motivos_selecionados: list = None,
    destacar_picos_auto: bool = True,
    sensibilidade_std: float = 0.5,
    mostrar_rotulos: bool = True
):
    df = df_imoveis.copy()
    df["Data de Baixa_dt"] = pd.to_datetime(df["Data de Baixa"], errors="coerce")
    baixados = df[df["Status"] == "Baixado"].dropna(subset=["Data de Baixa_dt"]).copy()
    baixados["Ano_Baixa"] = baixados["Data de Baixa_dt"].dt.year.astype(int)
    baixados["Mes_Baixa"] = baixados["Data de Baixa_dt"].dt.month.astype(int)

    ano_ini, ano_fim = intervalo_anos
    mes_ini, mes_fim = intervalo_meses
    baixados = baixados[
        (baixados["Ano_Baixa"] >= ano_ini) & (baixados["Ano_Baixa"] <= ano_fim) &
        (baixados["Mes_Baixa"] >= mes_ini) & (baixados["Mes_Baixa"] <= mes_fim)
    ]

    meses_nomes = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    faixa_m = list(range(mes_ini, mes_fim + 1))
    labels_m = [meses_nomes[m - 1] for m in faixa_m]

    if not motivos_selecionados:
        motivos_selecionados = [
            "Uso Próprio | Desistência | Reforma",
            "Adm. Particular | Locou Direto",
            "Vendeu o imóvel"
        ]

    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    cores = ["#f97316", "#10b981", "#6366f1", "#ef4444", "#06b6d4", "#eab308"]

    serie_acumulada = pd.Series(0, index=faixa_m, dtype=float)

    for idx, mot in enumerate(motivos_selecionados):
        saz = baixados[baixados["Motivo de Baixa"] == mot]["Mes_Baixa"].value_counts().reindex(faixa_m, fill_value=0)
        serie_acumulada = serie_acumulada.add(saz, fill_value=0)
        cor = cores[idx % len(cores)]
        ax.plot(labels_m, saz.values, marker="o", lw=2.3, markersize=5.5, color=cor, label=f"{mot} (n={int(saz.sum())})")
        if mostrar_rotulos:
            for x_i, val in enumerate(saz.values):
                if val > 0:
                    ax.annotate(str(int(val)), (x_i, val), xytext=(0, 4), textcoords="offset points",
                                ha="center", fontsize=8, fontweight="bold", color=cor)

    if destacar_picos_auto and not serie_acumulada.empty:
        _, _, picos_detectados = detectar_picos_automaticos(serie_acumulada, labels_m, sensibilidade_std)
        cores_faixas = ["#fdba74", "#c4b5fd", "#86efac", "#fca5a5", "#67e8f9"]
        if picos_detectados:
            for idx_p, (i_ini, i_fim, lbl_pico) in enumerate(picos_detectados):
                ax.axvspan(
                    max(0, i_ini - 0.25),
                    min(len(labels_m) - 1, i_fim + 0.25),
                    color=cores_faixas[idx_p % len(cores_faixas)],
                    alpha=0.28,
                    label=lbl_pico
                )
        else:
            ax.plot([], [], " ", label="Nenhum pico atípico acima do limiar")

    ax.set_title(f"Sazonalidade Mensal ({ano_ini}–{ano_fim} | {labels_m[0]} a {labels_m[-1]})", fontweight="bold")
    ax.set_xlabel("Mês da Baixa")
    ax.set_ylabel("Imóveis Retirados")
    ax.margins(y=0.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=True, fontsize=8)

    plt.tight_layout()
    return fig


def plot_a3_ranking_motivos(
    df_imoveis: pd.DataFrame,
    df_contratos: pd.DataFrame,
    fonte_dados: str = "Imóveis (Motivo de Baixa)",
    intervalo_anos: tuple = (2020, 2026),
    top_n: int = 8,
    exibir_em_percentual: bool = True
):
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ano_ini, ano_fim = intervalo_anos

    if "Contratos" in fonte_dados and not df_contratos.empty:
        df = df_contratos.copy()
        df["Dt"] = pd.to_datetime(df["Data de Rescisão"], errors="coerce")
        df = df.dropna(subset=["Dt"])
        df = df[(df["Dt"].dt.year >= ano_ini) & (df["Dt"].dt.year <= ano_fim)]
        col_mot = "Motivo de Rescisão"
        titulo_base = f"Top {top_n} Motivos de Rescisão de Inquilinos ({ano_ini}–{ano_fim})"
    else:
        df = df_imoveis[df_imoveis["Status"] == "Baixado"].copy()
        df["Dt"] = pd.to_datetime(df["Data de Baixa"], errors="coerce")
        df = df.dropna(subset=["Dt"])
        df = df[(df["Dt"].dt.year >= ano_ini) & (df["Dt"].dt.year <= ano_fim)]
        col_mot = "Motivo de Baixa"
        titulo_base = f"Top {top_n} Motivos de Evasão de Proprietários ({ano_ini}–{ano_fim})"

    contagem = df[col_mot].dropna().value_counts().head(top_n).sort_values(ascending=True)
    if contagem.empty:
        ax.text(0.5, 0.5, "Sem registros para os filtros selecionados", ha="center", va="center")
        return fig

    valores = (contagem / contagem.sum() * 100).round(1) if exibir_em_percentual else contagem
    sufixo = "%" if exibir_em_percentual else ""
    cores = sns.color_palette("viridis", len(valores))

    bars = ax.barh([textwrap.fill(formatar_rotulo(i), 30) for i in valores.index], valores.values, color=cores, edgecolor="black")
    for b in bars:
        w = b.get_width()
        ax.text(w + (valores.max() * 0.015), b.get_y() + b.get_height() / 2,
                f"{w:.1f}{sufixo}" if sufixo else f"{int(w)}", va="center", fontweight="bold", fontsize=8.5)

    ax.set_title(titulo_base, fontweight="bold")
    ax.set_xlabel("Participação (%)" if exibir_em_percentual else "Quantidade de Registros")
    ax.margins(x=0.18)
    plt.tight_layout()
    return fig