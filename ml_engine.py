import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)


def preparar_dataset_modelagem(df_imoveis: pd.DataFrame, df_contratos: pd.DataFrame, alvo_tipo: str, remover_outliers: bool = True):
    """
    Prepara a matriz X e o vetor y dinamicamente conforme a base atualizada no banco.
    alvo_tipo:
      - 'evasao_proprietario' (Classificação na base de Imóveis: Baixado=1 vs Alugado=0)
      - 'quebra_contrato' (Classificação no Merge Contratos+Imóveis: Quebra_Antecipada=1 vs 0)
      - 'preco_aluguel' (Regressão na base de Imóveis: Valor Líquido)
      - 'meses_locados' (Regressão no Merge Contratos+Imóveis: Meses_Locados)
    """
    if alvo_tipo in ["evasao_proprietario", "preco_aluguel"]:
        df = df_imoveis.copy()
        if alvo_tipo == "evasao_proprietario":
            df = df[df["Status"].isin(["Alugado", "Baixado"])].copy()
            df["Target"] = np.where(df["Status"] == "Baixado", 1, 0)
            features_num = ["Valor Líquido", "Taxa de Condominio", "Área Total (m2)", "Total de Quartos", "Vagas de Garagem", "Qtd_Imoveis_Proprietario"]
        else:
            df = df[df["Valor Líquido"] > 100].copy()
            df["Target"] = df["Valor Líquido"]
            features_num = ["Taxa de Condominio", "Área Total (m2)", "Total de Quartos", "Vagas de Garagem", "Qtd_Imoveis_Proprietario"]

        features_cat = ["Tipo", "Repasse Garantido", "Tx. Adm. Incide Sobre", "Bairro"]
    else:
        df_m = pd.merge(
            df_contratos, df_imoveis,
            left_on="Referência Imóvel", right_on="Referência",
            how="inner", suffixes=("_contrato", "_imovel")
        )
        if df_m.empty:
            return None, None, [], []
        df_m["Valor_Liquido_Final"] = df_m["Valor Líquido_contrato"].fillna(df_m.get("Valor Líquido_imovel", 0))
        df_m = df_m[df_m["Meses_Locados"] > 0].copy()
        df_m["Target"] = df_m["Quebra_Antecipada"] if alvo_tipo == "quebra_contrato" else df_m["Meses_Locados"]
        df = df_m
        features_num = ["Valor_Liquido_Final", "Taxa de Condominio", "Área Total (m2)", "Total de Quartos", "Vagas de Garagem", "Qtd_Imoveis_Proprietario"]
        features_cat = ["Garantia", "Tipo de Contrato", "Repasse Garantido"]

    features_num = [c for c in features_num if c in df.columns]
    features_cat = [c for c in features_cat if c in df.columns]

    # Cirurgia de Outliers via 1.5 * IQR (Aula 09 / Notebook anterior)
    if remover_outliers and len(df) > 20:
        col_filtro = "Target" if alvo_tipo in ["preco_aluguel", "meses_locados"] else features_num[0]
        q1 = df[col_filtro].quantile(0.25)
        q3 = df[col_filtro].quantile(0.75)
        iqr = q3 - q1
        df = df[(df[col_filtro] >= q1 - 1.5 * iqr) & (df[col_filtro] <= q3 + 1.5 * iqr)].copy()

    if "Bairro" in features_cat:
        top_bairros = df["Bairro"].value_counts().head(10).index
        df["Bairro"] = df["Bairro"].where(df["Bairro"].isin(top_bairros), "Outros_Bairros")

    df_model = df[features_num + features_cat + ["Target"]].dropna(subset=["Target"]).copy()
    for col in features_num:
        df_model[col] = df_model[col].fillna(df_model[col].median() if not df_model[col].dropna().empty else 0)

    df_encoded = pd.get_dummies(df_model, columns=features_cat, drop_first=True, dtype=int)
    X = df_encoded.drop(columns=["Target"])
    y = df_encoded["Target"]
    return X, y, features_num, list(X.columns)


def treinar_modelo_dinamico(X: pd.DataFrame, y: pd.Series, tipo_problema: str, algoritmo: str, params: dict) -> dict:
    """Treina o modelo escolhido pelo usuário em tempo real e avalia Treino vs Teste (Overfitting/Underfitting)."""
    test_size = params.get("test_size", 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    if tipo_problema == "classificacao":
        if algoritmo == "Árvore de Decisão":
            modelo = DecisionTreeClassifier(
                max_depth=params.get("max_depth", 5),
                min_samples_split=params.get("min_samples_split", 5),
                criterion=params.get("criterion", "gini"),
                random_state=42
            )
        elif algoritmo == "KNN (Vizinhos Mais Próximos)":
            modelo = KNeighborsClassifier(
                n_neighbors=params.get("n_neighbors", 5),
                weights=params.get("weights", "uniform")
            )
        elif algoritmo == "Random Forest":
            modelo = DecisionTreeClassifier(max_depth=params.get("max_depth", 6), random_state=42) if len(X_train) < 10 else RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 8),
                random_state=42
            )
        else:
            modelo = LogisticRegression(C=params.get("C", 1.0), max_iter=500, random_state=42)

        modelo.fit(X_train_sc, y_train)
        pred_train = modelo.predict(X_train_sc)
        pred_test = modelo.predict(X_test_sc)

        acc_train = accuracy_score(y_train, pred_train)
        acc_test = accuracy_score(y_test, pred_test)
        f1 = f1_score(y_test, pred_test, zero_division=0)
        cm = confusion_matrix(y_test, pred_test)

        gap = acc_train - acc_test
        if gap > 0.15:
            diagnostico = "Alerta de Overfitting: Acurácia de treino muito superior ao teste. Reduza a profundidade ou aumente K."
        elif acc_train < 0.60 and acc_test < 0.60:
            diagnostico = "Indício de Underfitting: Modelo simplificado demais para capturar o padrão."
        else:
            diagnostico = "Bom Ajuste (Generalização Adequada): Equilíbrio saudável entre treino e validação."

        return {
            "modelo": modelo,
            "scaler": scaler,
            "colunas_X": list(X.columns),
            "acc_train": acc_train,
            "acc_test": acc_test,
            "f1_test": f1,
            "confusion_matrix": cm,
            "diagnostico": diagnostico
        }

    else:
        if algoritmo == "Regressão Linear (OLS)":
            modelo = LinearRegression()
        elif algoritmo == "Regressão Ridge (Regularização L2)":
            modelo = Ridge(alpha=params.get("alpha", 1.0))
        elif algoritmo == "Árvore de Regressão":
            modelo = DecisionTreeRegressor(max_depth=params.get("max_depth", 5), random_state=42)
        else:
            modelo = RandomForestRegressor(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 8),
                random_state=42
            )

        modelo.fit(X_train_sc, y_train)
        pred_train = modelo.predict(X_train_sc)
        pred_test = modelo.predict(X_test_sc)

        r2_tr = r2_score(y_train, pred_train)
        r2_te = r2_score(y_test, pred_test)
        mae = mean_absolute_error(y_test, pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred_test))

        if (r2_tr - r2_te) > 0.20:
            diagnostico = "Alerta de Overfitting: O modelo decorou o treino e perdeu poder preditivo no teste."
        elif r2_te <= 0.05:
            diagnostico = "Underfitting / Sinal Linear Fraco: Próximo ao Baseline (R² ~ 0). Experimente Random Forest ou ative o filtro IQR."
        else:
            diagnostico = "Bom Ajuste: O modelo supera o baseline e generaliza bem em dados inéditos."

        return {
            "modelo": modelo,
            "scaler": scaler,
            "colunas_X": list(X.columns),
            "r2_train": r2_tr,
            "r2_test": r2_te,
            "mae": mae,
            "rmse": rmse,
            "y_test": y_test,
            "pred_test": pred_test,
            "diagnostico": diagnostico
        }