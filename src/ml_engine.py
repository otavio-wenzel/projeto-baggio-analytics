import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, confusion_matrix, silhouette_score
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

from src.schemas import formatar_rotulo


def preparar_matriz_features(df_limpo: pd.DataFrame):
    """Prepara matriz X (numéricas + One-Hot Encoding) para treinamento supervisionado."""
    df = df_limpo.copy()
    top_bairros = df["Bairro"].value_counts().head(8).index
    df["Bairro_Agrupado"] = df["Bairro"].where(df["Bairro"].isin(top_bairros), "Outros_Bairros")

    col_aluguel = "Valor Líquido_contrato" if "Valor Líquido_contrato" in df.columns else "Valor Líquido"
    features_num = [
        c for c in [
            col_aluguel, "Área Total (m2)", "Taxa Administrativa (%)",
            "Taxa de Condominio", "Qtd_Imoveis_Proprietario", "Meses Garantidos",
            "Total de Quartos", "Vagas de Garagem"
        ] if c in df.columns
    ]
    features_cat = [
        c for c in ["Garantia", "Tipo de Contrato", "Repasse Garantido", "Tx. Adm. Incide Sobre", "Bairro_Agrupado"]
        if c in df.columns
    ]

    df_model = df[features_num + features_cat].copy()
    for col in features_num:
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce").fillna(df_model[col].median())

    df_encoded = pd.get_dummies(df_model, columns=features_cat, drop_first=True, dtype=int)
    return df_encoded, features_num, features_cat


def instanciar_regressor(nome_alg: str, alpha: float = 1.0, max_depth: int = 5, n_estimators: int = 100):
    match nome_alg:
        case "Baseline (Média do Treino)":
            return DummyRegressor(strategy="mean")
        case "Regressão Linear (OLS)":
            return make_pipeline(StandardScaler(), LinearRegression())
        case "Regressão Ridge (Penalidade L2)":
            return make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=42))
        case "Regressão Lasso (Penalidade L1)":
            return make_pipeline(StandardScaler(), Lasso(alpha=alpha, random_state=42))
        case "Árvore de Decisão (Regressor)":
            return DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        case _:
            return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)


def diagnosticar_ajuste(rmse_treino: float, rmse_cv: float, r2_teste: float) -> str:
    if r2_teste <= 0.01:
        return "⚠️ Subajuste / Alta Variância (Ajuste α ou Profundidade)"
    if rmse_treino < 0.55 * rmse_cv:
        return "🚨 Indício de Sobreajuste (Reduza a Profundidade ou aumente α)"
    return "✅ Bom Ajuste (Equilíbrio entre Treino, Validação e Teste)"


def treinar_e_comparar_regressores(
    df_limpo: pd.DataFrame,
    algoritmo_principal: str = "Floresta Aleatória (Random Forest)",
    modelos_comparacao: list = None,
    alpha: float = 2.0,
    max_depth: int = 4,
    n_estimators: int = 100,
    test_size: float = 0.2
):
    X, _, _ = preparar_matriz_features(df_limpo)
    y = df_limpo["Meses_Locados"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    n_folds = min(5, max(2, len(X_train) // 4))
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    lista_modelos = ["Baseline (Média do Treino)", algoritmo_principal]
    if modelos_comparacao:
        for m in modelos_comparacao:
            if m not in lista_modelos:
                lista_modelos.append(m)

    linhas_tabela = []
    predicoes_teste = {}
    modelo_treinado_principal = None

    for nome in lista_modelos:
        mod = instanciar_regressor(nome, alpha, max_depth, n_estimators)
        scores_cv = -cross_val_score(mod, X_train, y_train, cv=kfold, scoring="neg_root_mean_squared_error")
        mod.fit(X_train, y_train)
        y_pred_tr = mod.predict(X_train)
        y_pred_te = mod.predict(X_test)
        predicoes_teste[nome] = y_pred_te

        rmse_tr = float(mean_squared_error(y_train, y_pred_tr) ** 0.5)
        rmse_te = float(mean_squared_error(y_test, y_pred_te) ** 0.5)
        mae_te = float(mean_absolute_error(y_test, y_pred_te))
        r2_te = float(r2_score(y_test, y_pred_te) * 100)

        linhas_tabela.append({
            "Modelo Avaliado": nome,
            "RMSE Validação (K-Fold)": round(float(scores_cv.mean()), 2),
            "Desvio K-Fold (±)": round(float(scores_cv.std()), 2),
            "Erro Médio (MAE Meses)": round(mae_te, 2),
            "Raiz do Erro (RMSE Meses)": round(rmse_te, 2),
            "Coeficiente R² (%)": round(r2_te, 2),
            "Diagnóstico do Modelo": diagnosticar_ajuste(rmse_tr, float(scores_cv.mean()), r2_te / 100.0)
        })

        if nome == algoritmo_principal:
            modelo_treinado_principal = mod

    df_comparativo = pd.DataFrame(linhas_tabela)

    estimador_interno = (
        modelo_treinado_principal.steps[-1][1]
        if hasattr(modelo_treinado_principal, "steps")
        else modelo_treinado_principal
    )
    rotulos_limpos_x = [formatar_rotulo(c) for c in X.columns]
    if hasattr(estimador_interno, "coef_"):
        pesos = pd.DataFrame({"Variável": rotulos_limpos_x, "Impacto": estimador_interno.coef_})
    elif hasattr(estimador_interno, "feature_importances_"):
        pesos = pd.DataFrame({"Variável": rotulos_limpos_x, "Impacto": estimador_interno.feature_importances_})
    else:
        pesos = pd.DataFrame({"Variável": rotulos_limpos_x, "Impacto": np.zeros(len(X.columns))})
    pesos = pesos.sort_values(by="Impacto", ascending=True)

    fig_diag, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    y_p_main = predicoes_teste[algoritmo_principal]
    y_p_base = predicoes_teste["Baseline (Média do Treino)"]

    axes[0].scatter(y_test, y_p_main, alpha=0.85, color="#2563eb", s=55, edgecolor="black", label=algoritmo_principal)
    axes[0].scatter(y_test, y_p_base, alpha=0.65, color="#6b7280", marker="X", s=45, label="Baseline (Média)")
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r", lw=1.8, label="Ajuste Ideal (y = ŷ)")
    axes[0].set_title("Real vs. Predito (Conjunto de Teste)", fontweight="bold")
    axes[0].set_xlabel("Tempo Real de Permanência (Meses)")
    axes[0].set_ylabel("Tempo Predito (Meses)")
    axes[0].legend(fontsize=8)

    top_p = pd.concat([pesos.head(5), pesos.tail(5)]).drop_duplicates()
    cores_p = ["#dc2626" if v < 0 else "#10b981" for v in top_p["Impacto"]]
    axes[1].barh(top_p["Variável"], top_p["Impacto"], color=cores_p, edgecolor="black")
    axes[1].axvline(0, color="black", lw=1.1)
    axes[1].set_title(f"Impacto das Variáveis ({algoritmo_principal})", fontweight="bold")
    plt.tight_layout()

    return modelo_treinado_principal, X.columns.tolist(), df_comparativo, fig_diag


def treinar_classificador_configuravel(
    df_limpo: pd.DataFrame,
    alvo_nome: str = "Quebra_Antecipada",
    algoritmo: str = "Floresta Aleatória (Classificador)",
    param_c_ou_k: float = 3.0,
    max_depth: int = 5,
    n_estimators: int = 100,
    test_size: float = 0.2
):
    X, _, _ = preparar_matriz_features(df_limpo)
    y = df_limpo[alvo_nome].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

    match algoritmo:
        case "Regressão Logística":
            clf = LogisticRegression(C=float(param_c_ou_k), max_iter=500, random_state=42)
        case "Vizinhos Mais Próximos (KNN)":
            k_viz = max(1, min(int(param_c_ou_k), len(X_train) - 1))
            clf = KNeighborsClassifier(n_neighbors=k_viz)
        case "Árvore de Decisão (Classificador)":
            clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        case _:
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig_cm, ax_cm = plt.subplots(figsize=(6.2, 3.0))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Permanece / Ativo (0)", "Evento de Risco (1)"],
        yticklabels=["Permanece / Ativo (0)", "Evento de Risco (1)"],
        ax=ax_cm
    )
    ax_cm.set_title(f"Matriz de Confusão — {formatar_rotulo(alvo_nome)}", fontweight="bold")
    ax_cm.set_xlabel("Classe Prevista pelo Modelo")
    ax_cm.set_ylabel("Classe Real Observada")
    plt.tight_layout()

    metricas = {
        "Acuracia_Pct": float(accuracy_score(y_test, y_pred) * 100),
        "F1_Score_Pct": float(f1_score(y_test, y_pred, zero_division=0) * 100),
        "Figura_CM": fig_cm
    }
    return clf, scaler, X.columns.tolist(), metricas


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