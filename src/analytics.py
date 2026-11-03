import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def preparar_base_cruzada(df_imoveis: pd.DataFrame, df_contratos: pd.DataFrame):
    """
    Realiza o Merge entre Contratos e Imóveis, calcula Meses_Locados,
    Quebra_Antecipada, Proprietario_Evadiu e aplica a Cirurgia de Outliers (1.5 x IQR).
    """
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

    col_status_imovel = "Status_imovel" if "Status_imovel" in df_merged.columns else "Status"
    df_merged["Proprietario_Evadiu"] = np.where(df_merged[col_status_imovel] == "Baixado", 1, 0)

    df_valid = df_merged[df_merged["Meses_Locados"] > 0].copy()
    if df_valid.empty:
        return df_valid, df_valid, 0.0

    q1 = df_valid["Meses_Locados"].quantile(0.25)
    q3 = df_valid["Meses_Locados"].quantile(0.75)
    iqr = q3 - q1
    limite_superior = float(q3 + 1.5 * iqr)

    df_limpo = df_valid[df_valid["Meses_Locados"] <= limite_superior].copy()
    return df_valid, df_limpo, limite_superior


def plot_escalada_e_sazonalidade(df_imoveis: pd.DataFrame):
    """Gera os Gráficos 1 e 2: Escalada Anual e Sazonalidade Mensal das Baixas."""
    df = df_imoveis.copy()
    df["Data de Baixa"] = pd.to_datetime(df["Data de Baixa"], errors="coerce")
    baixados = df[df["Status"] == "Baixado"].copy()
    baixados["Ano_Baixa"] = baixados["Data de Baixa"].dt.year
    baixados["Mes_Baixa"] = baixados["Data de Baixa"].dt.month

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    # Painel 1: Escalada Anual
    evol_anual = baixados[baixados["Ano_Baixa"] >= 2022]["Ano_Baixa"].value_counts().sort_index()
    anos_labels = [str(int(a)) for a in evol_anual.index]
    bars = axes[0].bar(anos_labels, evol_anual.values, color="#313695", edgecolor="black", width=0.55)
    axes[0].set_title("Escalada Anual de Retirada de Imóveis (>= 2022)", fontweight="bold")
    axes[0].set_xlabel("Ano da Baixa")
    axes[0].set_ylabel("Imóveis Baixados")
    for b in bars:
        axes[0].annotate(
            f"{int(b.get_height())}",
            (b.get_x() + b.get_width() / 2, b.get_height()),
            xytext=(0, 4), textcoords="offset points", ha="center", fontweight="bold"
        )

    # Painel 2: Sazonalidade Mensal
    meses_nomes = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    motivos = [
        ("Uso Próprio | Desistência | Reforma", "#d95f02", "o"),
        ("Adm. Particular | Locou Direto", "#1b9e77", "s"),
        ("Vendeu o imóvel", "#7570b3", "^")
    ]
    for motivo, cor, marcador in motivos:
        saz = baixados[baixados["Motivo de Baixa"] == motivo]["Mes_Baixa"].value_counts().reindex(range(1, 13), fill_value=0)
        axes[1].plot(meses_nomes, saz.values, marker=marcador, lw=2.5, color=cor, label=motivo)

    axes[1].axvspan(2, 5, color="#fee0b6", alpha=0.4, label="Pico Outono (Mar-Jun)")
    axes[1].axvspan(9, 10, color="#d8daeb", alpha=0.4, label="Pico Fim de Ano (Out-Nov)")
    axes[1].set_title("Sazonalidade Mensal das Retiradas por Motivo", fontweight="bold")
    axes[1].set_xlabel("Mês da Baixa")
    axes[1].set_ylabel("Imóveis Retirados")
    axes[1].legend(fontsize=8.5)

    plt.tight_layout()
    return fig


def plot_fatores_internos_e_carteira(df_imoveis: pd.DataFrame):
    """Gera os Gráficos de Repasse Garantido e Tamanho da Carteira do Proprietário."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    w = 0.35

    # Painel 1: Repasse Garantido
    ct_rep = pd.crosstab(df_imoveis["Repasse Garantido"], df_imoveis["Status"])
    ct_rep["Total"] = ct_rep.sum(axis=1)
    ct_rep = ct_rep[ct_rep["Total"] >= 10]
    if "Baixado" in ct_rep.columns and "Alugado" in ct_rep.columns:
        ct_rep["Pct_Baixado"] = (ct_rep["Baixado"] / ct_rep["Total"] * 100).round(1)
        ct_rep["Pct_Alugado"] = (ct_rep["Alugado"] / ct_rep["Total"] * 100).round(1)
        x_r = np.arange(len(ct_rep))
        axes[0].bar(x_r - w/2, ct_rep["Pct_Alugado"], w, label="% Alugado", color="#2b83ba", edgecolor="black")
        axes[0].bar(x_r + w/2, ct_rep["Pct_Baixado"], w, label="% Baixado", color="#d7191c", edgecolor="black")
        axes[0].set_xticks(x_r)
        axes[0].set_xticklabels(ct_rep.index)
        axes[0].set_title("Retenção vs. Evasão por Política de Repasse", fontweight="bold")
        axes[0].set_ylabel("Percentual (%)")
        axes[0].legend()

    # Painel 2: Porte da Carteira do Proprietário (Variável Reciclada LGPD)
    if "Faixa_Carteira_Prop" in df_imoveis.columns:
        ct_cart = pd.crosstab(df_imoveis["Faixa_Carteira_Prop"], df_imoveis["Status"])
        ct_cart["Total"] = ct_cart.sum(axis=1)
        if "Baixado" in ct_cart.columns and "Alugado" in ct_cart.columns:
            ct_cart["Pct_Baixado"] = (ct_cart["Baixado"] / ct_cart["Total"] * 100).round(1)
            ct_cart["Pct_Alugado"] = (ct_cart["Alugado"] / ct_cart["Total"] * 100).round(1)
            x_c = np.arange(len(ct_cart))
            axes[1].bar(x_c - w/2, ct_cart["Pct_Alugado"], w, label="% Alugado", color="#4daf4a", edgecolor="black")
            axes[1].bar(x_c + w/2, ct_cart["Pct_Baixado"], w, label="% Baixado", color="#984ea3", edgecolor="black")
            axes[1].set_xticks(x_c)
            axes[1].set_xticklabels(ct_cart.index, rotation=12)
            axes[1].set_title("Evasão por Porte do Investidor (Hash SHA-256)", fontweight="bold")
            axes[1].set_ylabel("Percentual (%)")
            axes[1].legend()

    plt.tight_layout()
    return fig


def plot_outliers_e_heatmap(df_valid: pd.DataFrame, df_limpo: pd.DataFrame, limite_superior: float):
    """Gera o Boxplot Antes/Depois do IQR e o Heatmap de Correlação de Pearson."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(x=df_valid["Meses_Locados"], color="#f46d43", ax=axes[0])
    axes[0].axvline(limite_superior, color="darkred", linestyle="--", lw=2, label=f"Limite IQR ({limite_superior:.1f}m)")
    axes[0].set_title(f"Antes do IQR (n={len(df_valid)}) vs. Corte ({len(df_limpo)} limpos)", fontweight="bold")
    axes[0].set_xlabel("Meses Locados")
    axes[0].legend()

    col_aluguel = "Valor Líquido_contrato" if "Valor Líquido_contrato" in df_limpo.columns else "Valor Líquido"
    cols_corr = [
        c for c in ["Meses_Locados", col_aluguel, "Área Total (m2)", "Total de Quartos",
                    "Taxa de Condominio", "Qtd_Imoveis_Proprietario", "Quebra_Antecipada"]
        if c in df_limpo.columns
    ]
    matriz_corr = df_limpo[cols_corr].corr()
    sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.8, ax=axes[1])
    axes[1].set_title("Matriz de Correlação Linear de Pearson", fontweight="bold")

    plt.tight_layout()
    return fig