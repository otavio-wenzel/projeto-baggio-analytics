import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.schemas import formatar_rotulo


def configurar_estilo_grafico(tema: str = "Claro"):
    """Configura cores e fontes proporcionais do Matplotlib/Seaborn conforme o tema."""
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
    """Formata os rótulos sem underscores, com quebra de linha e alinhamento seguro."""
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
    """
    Detecta automaticamente 0, 1, 2, 3 ou mais períodos de pico sazonal onde o volume
    mensal supera a Média + (sensibilidade_std * Desvio Padrão).
    """
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

    # Agrupa meses adjacentes em janelas contínuas de pico
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
    """Realiza o cruzamento relacional e aplica o corte de Outliers (Q3 + k * IQR)."""
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


# =====================================================================
# ANÁLISE 1: ESCALADA ANUAL DE RETIRADAS
# =====================================================================
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


# =====================================================================
# ANÁLISE 2: SAZONALIDADE MENSAL COM DETECÇÃO AUTOMÁTICA DE PICOS
# =====================================================================
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

    # Detecção Estatística Automática de Picos (0, 1, 2, 3 ou mais períodos)
    picos_detectados = []
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


# =====================================================================
# ANÁLISE 3: RANKING DOS MOTIVOS DE BAIXA OU RESCISÃO
# =====================================================================
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


# =====================================================================
# ANÁLISE 4: IMPACTO DO REPASSE GARANTIDO
# =====================================================================
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


# =====================================================================
# ANÁLISE 5: REGRA DE INCIDÊNCIA DA TAXA ADMINISTRATIVA
# =====================================================================
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


# =====================================================================
# ANÁLISE 6: PORTE DO PROPRIETÁRIO / CARTEIRA SHA-256
# =====================================================================
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


# =====================================================================
# ANÁLISE 7: GEOGRAFIA DE EVASÃO E VALORES POR BAIRRO
# =====================================================================
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


# =====================================================================
# ANÁLISE 8: GARANTIAS, SANEAMENTO IQR & CORRELAÇÃO
# =====================================================================
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


# =====================================================================
# ANÁLISE 9: CONSTRUTOR LIVRE COM RÓTULOS LIMPOS (SEM UNDERSCORE)
# =====================================================================
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