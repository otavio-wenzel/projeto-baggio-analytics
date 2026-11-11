"""
Camada Controller: Análises de Políticas Internas, Porte da Carteira,
Geografia de Bairros, Garantias/IQR e Construtor Livre (A4 a A9).
"""

import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.schemas import formatar_rotulo
from src.controllers.data_processor import formatar_eixo_x, preparar_base_cruzada


def plot_a4_repasse_garantido(
    df_imoveis: pd.DataFrame,
    faixa_aluguel: tuple = (0.0, 50000.0),
    tipos_selecionados: list = None,
    modo_exibicao: str = "Percentual Relativo (%)"
):
    df = df_imoveis.copy()
    df["Valor Líquido_num"] = pd.to_numeric(df["Valor Líquido"], errors="coerce").fillna(0)
    df = df[(df["Valor Líquido_num"] >= faixa_aluguel[0]) & (df["Valor Líquido_num"] <= faixa_aluguel[1])]
    if tipos_selecionados:
        df = df[df["Tipo"].isin(tipos_selecionados)]

    fig, ax = plt.subplots(figsize=(7.8, 3.3))
    ct = pd.crosstab(df["Repasse Garantido"], df["Status"])
    ct["Total"] = ct.sum(axis=1)
    ct = ct[ct["Total"] >= 5]

    if ct.empty or "Baixado" not in ct.columns or "Alugado" not in ct.columns:
        ax.text(0.5, 0.5, "Dados insuficientes para o filtro selecionado", ha="center", va="center")
        return fig

    if modo_exibicao == "Percentual Relativo (%)":
        v_alug = (ct["Alugado"] / ct["Total"] * 100).round(1)
        v_baix = (ct["Baixado"] / ct["Total"] * 100).round(1)
        sufixo = "%"
        ylabel = "Percentual da Carteira (%)"
    else:
        v_alug = ct["Alugado"]
        v_baix = ct["Baixado"]
        sufixo = ""
        ylabel = "Quantidade de Imóveis"

    x = np.arange(len(ct))
    w = 0.34
    b1 = ax.bar(x - w / 2, v_alug, w, label="Alugado (Retido)", color="#2563eb", edgecolor="black")
    b2 = ax.bar(x + w / 2, v_baix, w, label="Baixado (Evasão)", color="#dc2626", edgecolor="black")

    for r in list(b1) + list(b2):
        h = r.get_height()
        ax.annotate(f"{h:.1f}{sufixo}" if sufixo else f"{int(h)}", (r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold", fontsize=8.5)

    formatar_eixo_x(ax, ct.index, largura_max=22, rotacao=12)
    ax.set_title(f"Retenção vs. Evasão por Política de Repasse (R$ {faixa_aluguel[0]:.0f}–R$ {faixa_aluguel[1]:.0f})", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.margins(y=0.2)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_a5_taxa_administrativa(
    df_imoveis: pd.DataFrame,
    min_imoveis_regra: int = 15,
    metrica_analise: str = "Taxa de Evasão (% Baixado)"
):
    fig, ax = plt.subplots(figsize=(8.0, 3.3))
    df = df_imoveis.copy()
    df["Valor Líquido_num"] = pd.to_numeric(df["Valor Líquido"], errors="coerce")

    ct = pd.crosstab(df["Tx. Adm. Incide Sobre"], df["Status"])
    ct["Total"] = ct.sum(axis=1)
    ct = ct[ct["Total"] >= min_imoveis_regra].copy()

    if ct.empty:
        ax.text(0.5, 0.5, "Nenhuma categoria atinge o mínimo de imóveis selecionado", ha="center", va="center")
        return fig

    if metrica_analise == "Taxa de Evasão (% Baixado)" and "Baixado" in ct.columns:
        serie = (ct["Baixado"] / ct["Total"] * 100).round(1).sort_values(ascending=True)
        sufixo = "%"
        cor_barras = ["#10b981" if v < 32 else "#f59e0b" if v < 38 else "#dc2626" for v in serie.values]
    else:
        medias = df[df["Tx. Adm. Incide Sobre"].isin(ct.index)].groupby("Tx. Adm. Incide Sobre")["Valor Líquido_num"].mean().round(0)
        serie = medias.sort_values(ascending=True)
        sufixo = " R$"
        cor_barras = sns.color_palette("Blues_r", len(serie))

    bars = ax.barh([textwrap.fill(formatar_rotulo(i), 28) for i in serie.index], serie.values, color=cor_barras, edgecolor="black", height=0.55)
    for b in bars:
        w = b.get_width()
        ax.text(w + (serie.max() * 0.015), b.get_y() + b.get_height() / 2, f"{w:.1f}{sufixo}", va="center", fontweight="bold", fontsize=8.5)

    ax.set_title(f"Base da Taxa Administrativa — {metrica_analise}", fontweight="bold")
    ax.set_xlabel(metrica_analise)
    ax.margins(x=0.18)
    plt.tight_layout()
    return fig


def plot_a6_porte_carteira(
    df_imoveis: pd.DataFrame,
    faixas_selecionadas: list = None,
    metrica: str = "Retenção vs. Evasão (%)"
):
    fig, ax = plt.subplots(figsize=(7.8, 3.3))
    df = df_imoveis.copy()
    if "Faixa_Carteira_Prop" not in df.columns:
        ax.text(0.5, 0.5, "Porte da carteira não disponível", ha="center", va="center")
        return fig

    if faixas_selecionadas:
        df = df[df["Faixa_Carteira_Prop"].isin(faixas_selecionadas)]

    if metrica == "Retenção vs. Evasão (%)":
        ct = pd.crosstab(df["Faixa_Carteira_Prop"], df["Status"])
        ct["Total"] = ct.sum(axis=1)
        if "Alugado" in ct.columns and "Baixado" in ct.columns:
            alug = (ct["Alugado"] / ct["Total"] * 100).round(1)
            baix = (ct["Baixado"] / ct["Total"] * 100).round(1)
            x = np.arange(len(ct))
            w = 0.34
            b1 = ax.bar(x - w / 2, alug, w, label="% Alugado (Ativo)", color="#10b981", edgecolor="black")
            b2 = ax.bar(x + w / 2, baix, w, label="% Baixado (Evasão)", color="#8b5cf6", edgecolor="black")
            for r in list(b1) + list(b2):
                h = r.get_height()
                ax.annotate(f"{h:.1f}%", (r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontweight="bold", fontsize=8.5)
            formatar_eixo_x(ax, ct.index, largura_max=18, rotacao=12)
            ax.set_ylabel("Percentual (%)")
            ax.margins(y=0.2)
            ax.legend()
    else:
        col_num = "Valor Líquido" if "Aluguel" in metrica else "Taxa de Condominio"
        df[col_num] = pd.to_numeric(df[col_num], errors="coerce")
        q95 = df[col_num].quantile(0.95)
        df_sub = df[df[col_num] <= q95]
        sns.barplot(data=df_sub, x="Faixa_Carteira_Prop", y=col_num, palette="viridis", edgecolor="black", capsize=0.1, ax=ax)
        formatar_eixo_x(ax, df_sub["Faixa_Carteira_Prop"].unique(), largura_max=18, rotacao=12)
        ax.set_xlabel("Porte da Carteira do Proprietário")
        ax.set_ylabel(formatar_rotulo(col_num))

    ax.set_title(f"Porte da Carteira do Investidor — {metrica}", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_a7_geografia_bairros(
    df_imoveis: pd.DataFrame,
    top_n_bairros: int = 12,
    metrica_bairro: str = "Taxa de Evasão (% Baixado)",
    ordenacao: str = "Maior para Menor"
):
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    df = df_imoveis.copy()
    top_b = df["Bairro"].value_counts().head(top_n_bairros).index
    df_b = df[df["Bairro"].isin(top_b)].copy()

    asc = True if ordenacao == "Maior para Menor" else False

    if metrica_bairro == "Taxa de Evasão (% Baixado)":
        ct = pd.crosstab(df_b["Bairro"], df_b["Status"])
        ct["Total"] = ct.sum(axis=1)
        serie = (ct["Baixado"] / ct["Total"] * 100).round(1).sort_values(ascending=asc)
        cores = ["#2563eb" if v < 30 else "#f59e0b" if v < 45 else "#dc2626" for v in serie.values]
        bars = ax.barh(serie.index, serie.values, color=cores, edgecolor="black")
        ax.axvline(35.0, color="#1e40af", linestyle="--", lw=1.6, label="Média da Carteira (35%)")
        for b, b_nome in zip(bars, serie.index):
            w = b.get_width()
            tot = int(ct.loc[b_nome, "Total"])
            ax.text(w + 0.8, b.get_y() + b.get_height() / 2, f"{w:.1f}% (n={tot})", va="center", fontweight="bold", fontsize=8)
        ax.set_xlabel("Taxa de Evasão (% de Imóveis Baixados)")
        ax.legend(loc="lower right")
    else:
        col_val = "Valor Líquido" if "Aluguel" in metrica_bairro else "Taxa de Condominio"
        df_b[col_val] = pd.to_numeric(df_b[col_val], errors="coerce")
        serie = df_b.groupby("Bairro")[col_val].median().round(0).sort_values(ascending=asc)
        bars = ax.barh(serie.index, serie.values, color=sns.color_palette("crest", len(serie)), edgecolor="black")
        for b in bars:
            w = b.get_width()
            ax.text(w + (serie.max() * 0.015), b.get_y() + b.get_height() / 2, f"R$ {w:.0f}", va="center", fontweight="bold", fontsize=8)
        ax.set_xlabel(f"Mediana de {formatar_rotulo(col_val)}")

    ax.set_title(f"Diagnóstico nos {top_n_bairros} Maiores Bairros — {metrica_bairro}", fontweight="bold")
    ax.margins(x=0.2)
    plt.tight_layout()
    return fig


def plot_a8_garantias_iqr_corr(
    df_imoveis: pd.DataFrame,
    df_contratos: pd.DataFrame,
    multiplicador_iqr: float = 1.5,
    intervalo_meses_filtro: tuple = (1.0, 200.0),
    modo_visual: str = "Permanência Média por Garantia"
):
    df_valid, df_limpo, limite_iqr = preparar_base_cruzada(df_imoveis, df_contratos, multiplicador_iqr)
    fig, ax = plt.subplots(figsize=(8.0, 3.4))

    if df_limpo.empty:
        ax.text(0.5, 0.5, "Envie Imóveis e Contratos para visualizar o cruzamento", ha="center", va="center")
        return fig

    df_limpo = df_limpo[
        (df_limpo["Meses_Locados"] >= intervalo_meses_filtro[0]) &
        (df_limpo["Meses_Locados"] <= intervalo_meses_filtro[1])
    ]

    if modo_visual == "Auditoria Boxplot (Antes vs. Depois do IQR)":
        fig.clf()
        fig.set_size_inches(9.2, 3.3)
        axes = fig.subplots(1, 2)
        sns.boxplot(x=df_valid["Meses_Locados"], color="#f97316", ax=axes[0])
        axes[0].axvline(limite_iqr, color="#dc2626", linestyle="--", lw=1.8, label=f"Cerca {multiplicador_iqr}×IQR ({limite_iqr:.1f}m)")
        axes[0].set_title(f"1. Base Bruta ({len(df_valid)} contratos)", fontweight="bold")
        axes[0].set_xlabel("Tempo de Permanência (Meses)")
        axes[0].legend()

        sns.boxplot(x=df_limpo["Meses_Locados"], color="#10b981", ax=axes[1])
        axes[1].set_title(f"2. Base Saneada ({len(df_limpo)} contratos)", fontweight="bold")
        axes[1].set_xlabel("Tempo de Permanência (Meses)")
    elif modo_visual == "Mapa de Calor (Correlação de Pearson)":
        col_alug = "Valor Líquido_contrato" if "Valor Líquido_contrato" in df_limpo.columns else "Valor Líquido"
        cols_c = [c for c in ["Meses_Locados", col_alug, "Área Total (m2)", "Total de Quartos", "Taxa de Condominio", "Qtd_Imoveis_Proprietario", "Quebra_Antecipada"] if c in df_limpo.columns]
        corr = df_limpo[cols_c].corr()
        rotulos_limpos = [formatar_rotulo(c) for c in corr.columns]
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=rotulos_limpos, yticklabels=rotulos_limpos, linewidths=0.8, ax=ax)
        formatar_eixo_x(ax, rotulos_limpos, largura_max=16, rotacao=25)
        ax.set_title("Matriz de Correlação Linear de Pearson (Base Saneada)", fontweight="bold")
    else:
        gar_disp = df_limpo["Garantia"].dropna().value_counts().index.tolist()
        sns.barplot(data=df_limpo, x="Garantia", y="Meses_Locados", order=gar_disp, palette="Blues_d", capsize=0.12, edgecolor="black", ax=ax)
        formatar_eixo_x(ax, gar_disp, largura_max=18, rotacao=15)
        ax.set_title(f"Permanência Média por Modalidade de Garantia (Corte {multiplicador_iqr}×IQR)", fontweight="bold")
        ax.set_xlabel("Modalidade de Garantia")
        ax.set_ylabel("Tempo de Permanência (Meses)")

    plt.tight_layout()
    return fig


def plot_a9_construtor_customizavel(
    df: pd.DataFrame,
    eixo_x: str,
    eixo_y: str,
    tipo_plot: str = "Barras Verticais (Média)",
    percentil_corte_y: int = 95,
    top_categorias_x: int = 10
):
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    df_sub = df[[eixo_x, eixo_y]].dropna().copy()
    df_sub[eixo_y] = pd.to_numeric(df_sub[eixo_y], errors="coerce")
    df_sub = df_sub.dropna(subset=[eixo_y])

    if df_sub.empty:
        ax.text(0.5, 0.5, "Sem dados válidos para as variáveis escolhidas", ha="center", va="center")
        return fig

    if percentil_corte_y < 100:
        limite_p = df_sub[eixo_y].quantile(percentil_corte_y / 100.0)
        df_sub = df_sub[df_sub[eixo_y] <= limite_p]

    top_cats = df_sub[eixo_x].value_counts().head(top_categorias_x).index
    df_sub = df_sub[df_sub[eixo_x].isin(top_cats)]
    ordem = df_sub.groupby(eixo_x)[eixo_y].mean().sort_values(ascending=False).index.tolist()

    lbl_x = formatar_rotulo(eixo_x)
    lbl_y = formatar_rotulo(eixo_y)

    if tipo_plot == "Barras Horizontais (Média)":
        sns.barplot(data=df_sub, y=eixo_x, x=eixo_y, order=ordem, palette="viridis", edgecolor="black", capsize=0.1, ax=ax)
        ax.set_xlabel(lbl_y)
        ax.set_ylabel(lbl_x)
    elif tipo_plot == "Boxplot (Quartis)":
        sns.boxplot(data=df_sub, x=eixo_x, y=eixo_y, order=ordem, palette="Set2", ax=ax)
        formatar_eixo_x(ax, ordem, largura_max=16, rotacao=25)
        ax.set_xlabel(lbl_x)
        ax.set_ylabel(lbl_y)
    elif tipo_plot == "Dispersão (Scatter)":
        sns.scatterplot(data=df_sub, x=eixo_x, y=eixo_y, alpha=0.75, s=55, color="#2563eb", ax=ax)
        formatar_eixo_x(ax, ordem, largura_max=16, rotacao=25)
        ax.set_xlabel(lbl_x)
        ax.set_ylabel(lbl_y)
    elif tipo_plot == "Pizza (Proporção de X)":
        cont = df_sub[eixo_x].value_counts().head(6)
        ax.pie(cont.values, labels=[textwrap.fill(formatar_rotulo(i), 18) for i in cont.index], autopct="%1.0f%%", startangle=90)
    else:
        sns.barplot(data=df_sub, x=eixo_x, y=eixo_y, order=ordem, palette="viridis", edgecolor="black", capsize=0.1, ax=ax)
        formatar_eixo_x(ax, ordem, largura_max=16, rotacao=25)
        ax.set_xlabel(lbl_x)
        ax.set_ylabel(lbl_y)

    ax.set_title(f"{lbl_y} por {lbl_x} (Até P{percentil_corte_y})", fontweight="bold")
    plt.tight_layout()
    return fig