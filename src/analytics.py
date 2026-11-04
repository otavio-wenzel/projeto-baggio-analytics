import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def configurar_estilo_grafico(tema: str = "Claro"):
    """Ajusta dinamicamente o estilo visual do Matplotlib e Seaborn conforme o tema."""
    if tema == "Escuro":
        plt.style.use("dark_background")
        plt.rcParams.update({
            "axes.facecolor": "#1e293b",
            "figure.facecolor": "#0f172a",
            "axes.edgecolor": "#475569",
            "grid.color": "#334155",
            "text.color": "#f8fafc",
            "axes.labelcolor": "#f8fafc",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#cbd5e1"
        })
    else:
        plt.style.use("default")
        sns.set_theme(style="whitegrid")


def preparar_base_cruzada(
    df_imoveis: pd.DataFrame,
    df_contratos: pd.DataFrame,
    multiplicador_iqr: float = 1.5
):
    """Realiza o Merge e aplica o filtro de Outliers parametrizável (padrão 1.5 x IQR)."""
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


def plot_subgrupo_sazonalidade(
    df_imoveis: pd.DataFrame,
    ano_minimo: int = 2022,
    motivos_selecionados: list = None,
    tipo_grafico_anual: str = "Barras"
):
    """Subgrupo 1: Escalada Anual e Sazonalidade Mensal customizável."""
    df = df_imoveis.copy()
    df["Data de Baixa"] = pd.to_datetime(df["Data de Baixa"], errors="coerce")
    baixados = df[df["Status"] == "Baixado"].copy()
    baixados["Ano_Baixa"] = baixados["Data de Baixa"].dt.year
    baixados["Mes_Baixa"] = baixados["Data de Baixa"].dt.month

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5))

    evol = baixados[baixados["Ano_Baixa"] >= ano_minimo]["Ano_Baixa"].value_counts().sort_index()
    anos_lbl = [str(int(a)) for a in evol.index]

    if tipo_grafico_anual == "Pizza" and len(evol) > 0:
        axes[0].pie(evol.values, labels=anos_lbl, autopct="%1.0f%%", startangle=90)
        axes[0].set_title(f"Proporção de Baixas por Ano (>= {ano_minimo})", fontweight="bold")
    elif tipo_grafico_anual == "Linha":
        axes[0].plot(anos_lbl, evol.values, marker="o", lw=3, color="#3b82f6")
        axes[0].set_title(f"Evolução Anual de Retiradas (>= {ano_minimo})", fontweight="bold")
        axes[0].set_ylabel("Imóveis Baixados")
    else:
        bars = axes[0].bar(anos_lbl, evol.values, color="#3b82f6", edgecolor="black", width=0.55)
        axes[0].set_title(f"Escalada Anual de Retiradas (>= {ano_minimo})", fontweight="bold")
        axes[0].set_ylabel("Imóveis Baixados")
        for b in bars:
            axes[0].annotate(
                f"{int(b.get_height())}",
                (b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold"
            )

    meses_nomes = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    if not motivos_selecionados:
        motivos_selecionados = [
            "Uso Próprio | Desistência | Reforma",
            "Adm. Particular | Locou Direto",
            "Vendeu o imóvel"
        ]
    paleta = ["#f97316", "#10b981", "#8b5cf6", "#ef4444", "#06b6d4"]
    for idx, mot in enumerate(motivos_selecionados):
        saz = baixados[baixados["Motivo de Baixa"] == mot]["Mes_Baixa"].value_counts().reindex(range(1, 13), fill_value=0)
        axes[1].plot(meses_nomes, saz.values, marker="o", lw=2.5, color=paleta[idx % len(paleta)], label=mot[:32])

    axes[1].axvspan(2, 5, color="#fdba74", alpha=0.25, label="1º Pico (Mar-Jun)")
    axes[1].axvspan(9, 10, color="#c4b5fd", alpha=0.25, label="2º Pico (Out-Nov)")
    axes[1].set_title("Sazonalidade Mensal por Motivo de Baixa", fontweight="bold")
    axes[1].set_xlabel("Mês")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_subgrupo_politicas(
    df_imoveis: pd.DataFrame,
    variavel_politica: str = "Repasse Garantido",
    min_amostras: int = 10
):
    """Subgrupo 2: Impacto de Fatores Internos (Repasse Garantido vs Taxa Adm)."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5))
    w = 0.35

    ct = pd.crosstab(df_imoveis[variavel_politica], df_imoveis["Status"])
    ct["Total"] = ct.sum(axis=1)
    ct = ct[ct["Total"] >= min_amostras].sort_values(by="Total", ascending=False)

    if "Baixado" in ct.columns and "Alugado" in ct.columns:
        ct["Pct_Baixado"] = (ct["Baixado"] / ct["Total"] * 100).round(1)
        ct["Pct_Alugado"] = (ct["Alugado"] / ct["Total"] * 100).round(1)
        x = np.arange(len(ct))
        axes[0].bar(x - w/2, ct["Pct_Alugado"], w, label="% Alugado (Retido)", color="#2563eb", edgecolor="black")
        axes[0].bar(x + w/2, ct["Pct_Baixado"], w, label="% Baixado (Evasão)", color="#dc2626", edgecolor="black")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([str(i)[:25] for i in ct.index], rotation=12, fontsize=8.5)
        axes[0].set_title(f"Retenção vs. Evasão: {variavel_politica}", fontweight="bold")
        axes[0].set_ylabel("Percentual (%)")
        axes[0].legend()

    # Painel 2: Proporção Geral de Status (Gráfico de Pizza com autopct da Aula)
    status_top = df_imoveis["Status"].value_counts().head(4)
    axes[1].pie(
        status_top.values,
        labels=status_top.index,
        autopct="%1.0f%%",
        startangle=140,
        colors=["#2563eb", "#dc2626", "#f59e0b", "#10b981"]
    )
    axes[1].set_title("Composição dos 4 Principais Status na Carteira", fontweight="bold")

    plt.tight_layout()
    return fig


def plot_subgrupo_geografia_carteira(
    df_imoveis: pd.DataFrame,
    top_n_bairros: int = 12
):
    """Subgrupo 3: Análise Geográfica por Bairro e Porte da Carteira do Investidor."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    top_b = df_imoveis["Bairro"].value_counts().head(top_n_bairros).index
    ct_b = pd.crosstab(df_imoveis[df_imoveis["Bairro"].isin(top_b)]["Bairro"], df_imoveis["Status"])
    ct_b["Total"] = ct_b.sum(axis=1)
    if "Baixado" in ct_b.columns:
        ct_b["Pct_Baixado"] = (ct_b["Baixado"] / ct_b["Total"] * 100).round(1)
        ct_b = ct_b.sort_values(by="Pct_Baixado", ascending=True)
        cores = ["#2563eb" if v < 30 else "#f59e0b" if v < 45 else "#dc2626" for v in ct_b["Pct_Baixado"]]
        axes[0].barh(ct_b.index, ct_b["Pct_Baixado"], color=cores, edgecolor="black")
        axes[0].axvline(35.0, color="#1e40af", linestyle="--", lw=1.8, label="Média (35%)")
        axes[0].set_title(f"Taxa de Evasão (% Baixado) nos Top {top_n_bairros} Bairros", fontweight="bold")
        axes[0].set_xlabel("% Imóveis Baixados")
        axes[0].legend()

    if "Faixa_Carteira_Prop" in df_imoveis.columns:
        ct_c = pd.crosstab(df_imoveis["Faixa_Carteira_Prop"], df_imoveis["Status"])
        ct_c["Total"] = ct_c.sum(axis=1)
        if "Baixado" in ct_c.columns and "Alugado" in ct_c.columns:
            ct_c["Pct_Baixado"] = (ct_c["Baixado"] / ct_c["Total"] * 100).round(1)
            ct_c["Pct_Alugado"] = (ct_c["Alugado"] / ct_c["Total"] * 100).round(1)
            x_c = np.arange(len(ct_c))
            w = 0.35
            axes[1].bar(x_c - w/2, ct_c["Pct_Alugado"], w, label="% Alugado", color="#10b981", edgecolor="black")
            axes[1].bar(x_c + w/2, ct_c["Pct_Baixado"], w, label="% Baixado", color="#8b5cf6", edgecolor="black")
            axes[1].set_xticks(x_c)
            axes[1].set_xticklabels(ct_c.index, rotation=12, fontsize=8.5)
            axes[1].set_title("Retenção vs. Evasão por Porte do Investidor", fontweight="bold")
            axes[1].legend()

    plt.tight_layout()
    return fig


def plot_subgrupo_garantias_iqr(
    df_valid: pd.DataFrame,
    df_limpo: pd.DataFrame,
    limite_iqr: float
):
    """Subgrupo 4: Permanência por Garantia, Cirurgia IQR e Matriz de Pearson."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))

    gar_principais = ["Seguro Fiança", "Fiador", "Sem Garantia", "Carta Fiança", "Título de Capitalização"]
    df_gar = df_limpo[df_limpo["Garantia"].isin(gar_principais)].copy()
    if not df_gar.empty:
        sns.barplot(
            data=df_gar, x="Garantia", y="Meses_Locados", hue="Garantia",
            palette="Blues_d", capsize=0.12, edgecolor="black", legend=False, ax=axes[0]
        )
        axes[0].set_title("Permanência Média por Garantia (Pós-IQR)", fontweight="bold")
        axes[0].set_ylabel("Meses Locados")
        axes[0].tick_params(axis="x", rotation=12)

    sns.boxplot(x=df_valid["Meses_Locados"], color="#f97316", ax=axes[1])
    axes[1].axvline(limite_iqr, color="#dc2626", linestyle="--", lw=2.2, label=f"Corte IQR ({limite_iqr:.1f}m)")
    axes[1].set_title(f"Auditoria de Outliers: Bruta ({len(df_valid)}) vs Saneada ({len(df_limpo)})", fontweight="bold")
    axes[1].set_xlabel("Meses Locados")
    axes[1].legend()

    plt.tight_layout()
    return fig


def plot_explorador_customizavel(
    df: pd.DataFrame,
    eixo_x: str,
    eixo_y: str,
    tipo_plot: str = "Barras (Média)"
):
    """Subgrupo 5: Construtor de visualização customizada pelo usuário."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    df_sub = df[[eixo_x, eixo_y]].dropna().copy()

    if df_sub[eixo_x].nunique() > 12 and df_sub[eixo_x].dtype == "object":
        top_cats = df_sub[eixo_x].value_counts().head(10).index
        df_sub = df_sub[df_sub[eixo_x].isin(top_cats)]

    if tipo_plot == "Boxplot":
        sns.boxplot(data=df_sub, x=eixo_x, y=eixo_y, palette="Set2", ax=ax)
    elif tipo_plot == "Dispersão (Scatter)":
        sns.scatterplot(data=df_sub, x=eixo_x, y=eixo_y, alpha=0.75, s=65, color="#2563eb", ax=ax)
    elif tipo_plot == "Pizza (Contagem de X)":
        cont = df_sub[eixo_x].value_counts().head(6)
        ax.pie(cont.values, labels=cont.index, autopct="%1.0f%%", startangle=90)
    else:
        sns.barplot(data=df_sub, x=eixo_x, y=eixo_y, palette="viridis", edgecolor="black", ax=ax)

    ax.set_title(f"Análise Customizada: {eixo_y} por {eixo_x} ({tipo_plot})", fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    return fig