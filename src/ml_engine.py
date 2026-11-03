import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, confusion_matrix, silhouette_score
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage


def preparar_matriz_features(df_limpo: pd.DataFrame):
    """Prepara as features numéricas e categóricas (One-Hot Encoding) para ML Supervisionado."""
    df = df_limpo.copy()
    top_bairros = df["Bairro"].value_counts().head(8).index
    df["Bairro_Agrupado"] = df["Bairro"].where(df["Bairro"].isin(top_bairros), "Outros_Bairros")

    col_aluguel = "Valor Líquido_contrato" if "Valor Líquido_contrato" in df.columns else "Valor Líquido"
    features_num = [
        col_aluguel, "Área Total (m2)", "Taxa Administrativa (%)",
        "Taxa de Condominio", "Qtd_Imoveis_Proprietario", "Meses Garantidos",
        "Total de Quartos", "Vagas de Garagem"
    ]
    features_num = [c for c in features_num if c in df.columns]

    features_cat = ["Garantia", "Tipo de Contrato", "Repasse Garantido", "Tx. Adm. Incide Sobre", "Bairro_Agrupado"]
    features_cat = [c for c in features_cat if c in df.columns]

    df_model = df[features_num + features_cat].copy()
    for col in features_num:
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce")
        df_model[col] = df_model[col].fillna(df_model[col].median())

    df_encoded = pd.get_dummies(df_model, columns=features_cat, drop_first=True, dtype=int)
    return df_encoded, features_num, features_cat


def treinar_regressor_configuravel(
    df_limpo: pd.DataFrame,
    algoritmo: str = "LinearRegression (OLS)",
    alpha: float = 1.0,
    n_estimators: int = 100,
    test_size: float = 0.2
):
    """
    Treina dinamicamente o modelo de Regressão escolhido pelo usuário para prever Meses_Locados.
    """
    X, features_num, _ = preparar_matriz_features(df_limpo)
    y = df_limpo["Meses_Locados"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if algoritmo == "Ridge (Penalidade L2)":
        modelo = Ridge(alpha=alpha, random_state=42)
    elif algoritmo == "Lasso (Penalidade L1)":
        modelo = Lasso(alpha=alpha, random_state=42)
    elif algoritmo == "RandomForestRegressor":
        modelo = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    else:
        modelo = LinearRegression()

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    metricas = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2_Pct": float(r2_score(y_test, y_pred) * 100),
        "Amostras_Treino": len(X_train),
        "Amostras_Teste": len(X_test)
    }

    # Extração de Pesos Beta ou Importância das Variáveis
    if hasattr(modelo, "coef_"):
        pesos = pd.DataFrame({"Feature": X.columns, "Peso": modelo.coef_})
    else:
        pesos = pd.DataFrame({"Feature": X.columns, "Peso": modelo.feature_importances_})

    pesos = pesos.sort_values(by="Peso", ascending=True)
    return modelo, X.columns.tolist(), metricas, pesos


def treinar_classificador_configuravel(
    df_limpo: pd.DataFrame,
    alvo_nome: str = "Quebra_Antecipada",
    algoritmo: str = "RandomForestClassifier",
    param_c_ou_k: float = 1.0,
    n_estimators: int = 100,
    test_size: float = 0.2
):
    """
    Atende ao requisito de Classificadores Configuráveis pelo Usuário:
    permite prever Quebra_Antecipada (Inquilino) ou Proprietario_Evadiu (Status Baixado).
    """
    X, _, _ = preparar_matriz_features(df_limpo)
    y = df_limpo[alvo_nome].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )

    if algoritmo == "LogisticRegression":
        clf = LogisticRegression(C=float(param_c_ou_k), max_iter=500, random_state=42)
    elif algoritmo == "KNN (K-Nearest Neighbors)":
        k_vizinhos = max(1, min(int(param_c_ou_k), len(X_train) - 1))
        clf = KNeighborsClassifier(n_neighbors=k_vizinhos)
    else:
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    metricas = {
        "Acuracia_Pct": float(accuracy_score(y_test, y_pred) * 100),
        "F1_Score_Pct": float(f1_score(y_test, y_pred, zero_division=0) * 100),
        "Matriz_Confusao": confusion_matrix(y_test, y_pred)
    }
    return clf, scaler, X.columns.tolist(), metricas


def executar_pipeline_clusters(
    df_limpo: pd.DataFrame,
    n_clusters: int = 4,
    eps_dbscan: float = 1.1,
    min_samples_dbscan: int = 5
):
    """
    Executa K-Means com StandardScaler, Dendrograma de Ward, DBSCAN (-1) e Projeção PCA 2D.
    """
    col_aluguel = "Valor Líquido_contrato" if "Valor Líquido_contrato" in df_limpo.columns else "Valor Líquido"
    cols_cluster = [col_aluguel, "Meses_Locados", "Taxa de Condominio", "Qtd_Imoveis_Proprietario"]
    df_clust = df_limpo.dropna(subset=cols_cluster).copy()

    q1_v, q3_v = df_clust[col_aluguel].quantile([0.25, 0.75])
    df_clust = df_clust[df_clust[col_aluguel] <= (q3_v + 3 * (q3_v - q1_v))].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clust[cols_cluster])

    # 1. K-Means Padronizado
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clust["Cluster_KMeans"] = kmeans.fit_predict(X_scaled)
    sil_score = float(silhouette_score(X_scaled, df_clust["Cluster_KMeans"])) if len(df_clust) > n_clusters else 0.0

    # 2. DBSCAN para Caça a Anomalias (-1)
    dbscan = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan)
    df_clust["Cluster_DBSCAN"] = dbscan.fit_predict(X_scaled)

    # 3. Redução de Dimensionalidade PCA (4D -> 2D)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    df_clust["PC1"] = coords[:, 0]
    df_clust["PC2"] = coords[:, 1]
    var_explicada = pca.explained_variance_ratio_ * 100

    # Figura PCA Lado a Lado (K-Means vs DBSCAN)
    fig_pca, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    sns.scatterplot(
        data=df_clust, x="PC1", y="PC2", hue="Cluster_KMeans",
        palette="viridis", s=75, alpha=0.85, ax=axes[0]
    )
    axes[0].set_title(f"Mapa PCA 2D ({var_explicada.sum():.1f}% Var.) — {n_clusters} Clusters K-Means", fontweight="bold")

    normais = df_clust[df_clust["Cluster_DBSCAN"] != -1]
    anomalias = df_clust[df_clust["Cluster_DBSCAN"] == -1]
    if not normais.empty:
        sns.scatterplot(
            data=normais, x="PC1", y="PC2", hue="Cluster_DBSCAN",
            palette="Set2", s=70, alpha=0.8, ax=axes[1]
        )
    if not anomalias.empty:
        axes[1].scatter(
            anomalias["PC1"], anomalias["PC2"], color="crimson",
            marker="P", s=140, edgecolor="black", label=f"Anomalias (-1): {len(anomalias)}"
        )
    axes[1].set_title(f"DBSCAN (eps={eps_dbscan}) — Caça a Anomalias Operacionais", fontweight="bold")
    axes[1].legend()
    plt.tight_layout()

    # Figura Dendrograma de Ward
    fig_dendro, ax_d = plt.subplots(figsize=(12, 4.2))
    matriz_ward = linkage(X_scaled, method="ward")
    dendrogram(matriz_ward, truncate_mode="lastp", p=25, leaf_rotation=45, show_contracted=True, ax=ax_d)
    ax_d.set_title("Dendrograma Hierárquico Aglomerativo (Método de Ward)", fontweight="bold")
    ax_d.set_ylabel("Distância de Ward (Inércia WCSS)")
    plt.tight_layout()

    resumo_clusters = df_clust.groupby("Cluster_KMeans").agg(
        Contratos=("Cluster_KMeans", "count"),
        Aluguel_Medio=(col_aluguel, "mean"),
        Condominio_Medio=("Taxa de Condominio", "mean"),
        Meses_Locados_Medio=("Meses_Locados", "mean"),
        Imoveis_Por_Prop=("Qtd_Imoveis_Proprietario", "mean"),
        Quebra_Antecipada_Pct=("Quebra_Antecipada", lambda s: s.mean() * 100),
        Evasao_Prop_Pct=("Proprietario_Evadiu", lambda s: s.mean() * 100)
    ).round(1)

    return df_clust, resumo_clusters, anomalias, sil_score, fig_pca, fig_dendro