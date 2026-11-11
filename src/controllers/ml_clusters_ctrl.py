import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

from src.models.schemas import formatar_rotulo


def executar_pipeline_clusters(
    df_limpo: pd.DataFrame,
    n_clusters: int = 4,
    eps_dbscan: float = 1.1,
    min_samples_dbscan: int = 5
):
    col_aluguel = "Valor Líquido_contrato" if "Valor Líquido_contrato" in df_limpo.columns else "Valor Líquido"
    cols_cluster = [col_aluguel, "Meses_Locados", "Taxa de Condominio", "Qtd_Imoveis_Proprietario"]
    df_clust = df_limpo.dropna(subset=cols_cluster).copy()

    q1_v, q3_v = df_clust[col_aluguel].quantile([0.25, 0.75])
    df_clust = df_clust[df_clust[col_aluguel] <= (q3_v + 3 * (q3_v - q1_v))].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clust[cols_cluster])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clust["Grupo K-Means"] = kmeans.fit_predict(X_scaled)
    sil_score = float(silhouette_score(X_scaled, df_clust["Grupo K-Means"])) if len(df_clust) > n_clusters else 0.0

    dbscan = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan)
    df_clust["Densidade DBSCAN"] = dbscan.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    df_clust["Eixo PC1"], df_clust["Eixo PC2"] = coords[:, 0], coords[:, 1]
    var_exp = pca.explained_variance_ratio_ * 100

    fig_pca, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    sns.scatterplot(data=df_clust, x="Eixo PC1", y="Eixo PC2", hue="Grupo K-Means", palette="viridis", s=60, ax=axes[0])
    axes[0].set_title(f"Mapa PCA 2D ({var_exp.sum():.1f}% Var.) — {n_clusters} Grupos K-Means", fontweight="bold")

    normais = df_clust[df_clust["Densidade DBSCAN"] != -1]
    anomalias = df_clust[df_clust["Densidade DBSCAN"] == -1]
    if not normais.empty:
        sns.scatterplot(data=normais, x="Eixo PC1", y="Eixo PC2", hue="Densidade DBSCAN", palette="Set2", s=55, ax=axes[1])
    if not anomalias.empty:
        axes[1].scatter(anomalias["Eixo PC1"], anomalias["Eixo PC2"], color="crimson", marker="P", s=110, edgecolor="black", label=f"Anomalias (-1): {len(anomalias)}")
    axes[1].set_title(f"DBSCAN (Raio={eps_dbscan}) — Isolamento de Anomalias", fontweight="bold")
    axes[1].legend(fontsize=8)
    plt.tight_layout()

    fig_dendro, ax_d = plt.subplots(figsize=(8.2, 3.2))
    matriz_ward = linkage(X_scaled, method="ward")
    dendrogram(matriz_ward, truncate_mode="lastp", p=25, leaf_rotation=45, show_contracted=True, ax=ax_d)
    ax_d.set_title("Dendrograma Hierárquico Aglomerativo (Método de Ward)", fontweight="bold")
    ax_d.set_ylabel("Distância de Ward (Inércia WCSS)")
    plt.tight_layout()

    resumo_clusters = df_clust.groupby("Grupo K-Means").agg(
        Contratos=("Grupo K-Means", "count"),
        Aluguel_Medio=(col_aluguel, "mean"),
        Condominio_Medio=("Taxa de Condominio", "mean"),
        Meses_Medio=("Meses_Locados", "mean"),
        Imoveis_Prop=("Qtd_Imoveis_Proprietario", "mean"),
        Quebra_Pct=("Quebra_Antecipada", lambda s: s.mean() * 100),
        Evasao_Pct=("Proprietario_Evadiu", lambda s: s.mean() * 100)
    ).round(1)
    resumo_clusters.columns = [formatar_rotulo(c) for c in resumo_clusters.columns]

    return df_clust, resumo_clusters, anomalias, sil_score, fig_pca, fig_dendro