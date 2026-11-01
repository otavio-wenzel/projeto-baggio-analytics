import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from database import StorageRepository, carregar_configuracoes, salvar_configuracoes
from data_processor import (
    ler_arquivo_upload, higienizar_imoveis,
    higienizar_contratos, reconciliar_upsert
)
from ml_engine import preparar_dataset_modelagem, treinar_modelo_dinamico

st.set_page_config(
    page_title="Baggio Analytics — Gestão & Predição Imobiliária",
    page_icon="🏢",
    layout="wide"
)

sns.set_theme(style="whitegrid")

# Inicializa o repositório desacoplado
repo = StorageRepository()
config_atual = carregar_configuracoes()

# ================= BARRA LATERAL: CONFIGURAÇÃO DE BANCO DESACOPLADO =================
st.sidebar.title("⚙️ Arquitetura & Banco")
st.sidebar.caption("Camada de Persistência Desacoplada (settings.json)")

motor_selecionado = st.sidebar.selectbox(
    "Motor de Persistência Ativo:",
    options=["sqlite", "filesystem"],
    index=0 if config_atual.get("storage_engine") == "sqlite" else 1,
    format_func=lambda x: "SQLite3 Relacional (.db)" if x == "sqlite" else "FileSystem Atômico (.csv/.json)"
)

formato_fs = config_atual.get("filesystem_format", "csv")
if motor_selecionado == "filesystem":
    formato_fs = st.sidebar.radio(
        "Formato em Disco:",
        options=["csv", "json"],
        index=0 if formato_fs == "csv" else 1
    )

if st.sidebar.button("💾 Salvar Configuração de Banco"):
    config_atual["storage_engine"] = motor_selecionado
    config_atual["filesystem_format"] = formato_fs
    salvar_configuracoes(config_atual)
    st.sidebar.success(f"Motor alterado para {motor_selecionado.upper()}!")
    st.rerun()

# Carrega as bases a partir do motor ativo
df_imoveis = repo.carregar_tabela("imoveis")
df_contratos = repo.carregar_tabela("contratos")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Estado do Banco")
st.sidebar.metric("Imóveis Persistidos", len(df_imoveis))
st.sidebar.metric("Contratos Rescindidos", len(df_contratos))

# ================= CABEÇALHO E NAVEGAÇÃO POR ABAS =================
st.title("🏢 Sistema Inteligente de Gestão de Carteira e Predição Imobiliária")
st.markdown(
    "**Arquitetura Desacoplada com Reconciliação de Dados (Upsert), Análise Geográfica e Machine Learning Dinâmico**"
)

aba_ingestao, aba_eda, aba_mapa, aba_ml, aba_logs = st.tabs([
    "📥 1. Upload & Reconciliação (Upsert)",
    "📈 2. Análise Exploratória & Retenção",
    "🗺️ 3. Mapa Geográfico Interativo",
    "🤖 4. Machine Learning Dinâmico",
    "🛡️ 5. Auditoria & Logs (TXT)"
])

# ================= ABA 1: UPLOAD E RECONCILIAÇÃO (IMÓVEIS E CONTRATOS) =================
with aba_ingestao:
    st.subheader("Alimentação Contínua e Atualização de Bases (.csv, .json, .xml, .xlsx)")
    st.write(
        "Envie lotes novos ou planilhas atualizadas. O motor utiliza **Conjuntos (`set`)** para "
        "identificar entradas inéditas, atualizar o status de imóveis que saíram/entraram e impedir duplicatas."
    )

    col_up1, col_up2 = st.columns(2)

    with col_up1:
        st.markdown("#### 🏠 Base de Imóveis (Entradas, Saídas e Posse)")
        arq_imoveis = st.file_uploader(
            "Selecione arquivo de Imóveis (.csv, .json, .xml, .xlsx)",
            type=["csv", "json", "xml", "xlsx"],
            key="up_imoveis"
        )
        if arq_imoveis is not None:
            if st.button("🔄 Processar e Reconciliar Imóveis", type="primary"):
                try:
                    df_bruto = ler_arquivo_upload(arq_imoveis)
                    df_limpo = higienizar_imoveis(df_bruto)
                    df_consolidado, delta = reconciliar_upsert(df_imoveis, df_limpo, "Referência")
                    repo.salvar_tabela("imoveis", df_consolidado)
                    st.success("Base de Imóveis atualizada e sincronizada no banco!")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Novos (Entradas)", f"+{delta['novos_inseridos']}")
                    c2.metric("Atualizados", delta["atualizados"])
                    c3.metric("Mudanças de Status", delta["mudancas_status"])
                    c4.metric("Total na Base", delta["total_consolidado"])
                except Exception as e:
                    st.error(f"Erro ao processar arquivo de imóveis: {e}")

    with col_up2:
        st.markdown("#### 📄 Base de Contratos Rescindidos (Lotes Mensais)")
        arq_contratos = st.file_uploader(
            "Selecione arquivo de Contratos (.csv, .json, .xml, .xlsx)",
            type=["csv", "json", "xml", "xlsx"],
            key="up_contratos"
        )
        if arq_contratos is not None:
            if st.button("🔄 Processar e Reconciliar Contratos", type="primary"):
                try:
                    df_bruto_c = ler_arquivo_upload(arq_contratos)
                    df_limpo_c = higienizar_contratos(df_bruto_c)
                    df_consolidado_c, delta_c = reconciliar_upsert(df_contratos, df_limpo_c, "Referência Contrato")
                    repo.salvar_tabela("contratos", df_consolidado_c)
                    st.success("Base de Contratos atualizada sem duplicidades!")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Novas Rescisões", f"+{delta_c['novos_inseridos']}")
                    c2.metric("Contratos Atualizados", delta_c["atualizados"])
                    c3.metric("Total Acumulado", delta_c["total_consolidado"])
                except Exception as e:
                    st.error(f"Erro ao processar arquivo de contratos: {e}")

    st.markdown("---")
    st.markdown("#### 🔍 Inspeção Rápida das Tabelas Persistidas")
    escolha_preview = st.radio("Visualizar amostra de:", ["Imóveis Persistidos", "Contratos Persistidos"], horizontal=True)
    if escolha_preview == "Imóveis Persistidos" and not df_imoveis.empty:
        st.dataframe(df_imoveis.head(20), use_container_width=True)
    elif escolha_preview == "Contratos Persistidos" and not df_contratos.empty:
        st.dataframe(df_contratos.head(20), use_container_width=True)
    else:
        st.info("Carregue os arquivos acima para popular o banco de dados.")

# ================= ABA 2: ANÁLISE EXPLORATÓRIA E VISUALIZAÇÃO =================
with aba_eda:
    if df_imoveis.empty:
        st.warning("Realize o upload da base de Imóveis na Aba 1 para liberar os gráficos automáticos.")
    else:
        st.subheader("📊 Painel de Indicadores e Comportamento da Carteira")
        k1, k2, k3, k4 = st.columns(4)
        alugados = int((df_imoveis["Status"] == "Alugado").sum()) if "Status" in df_imoveis.columns else 0
        baixados = int((df_imoveis["Status"] == "Baixado").sum()) if "Status" in df_imoveis.columns else 0
        ticket_medio = df_imoveis["Valor Líquido"].median() if "Valor Líquido" in df_imoveis.columns else 0

        k1.metric("Total de Imóveis", len(df_imoveis))
        k2.metric("Imóveis Alugados (Ativos)", alugados)
        k3.metric("Imóveis Baixados (Evasão)", baixados)
        k4.metric("Mediana de Aluguel Líquido", f"R$ {ticket_medio:,.2f}")

        g_col1, g_col2 = st.columns(2)

        with g_col1:
            st.markdown("**Distribuição de Preços (Valor Líquido) nos 10 Maiores Bairros**")
            top10_b = df_imoveis["Bairro"].value_counts().head(10).index
            df_top_b = df_imoveis[(df_imoveis["Bairro"].isin(top10_b)) & (df_imoveis["Valor Líquido"] <= 15000)]
            fig1, ax1 = plt.subplots(figsize=(8, 4.5))
            sns.boxplot(data=df_top_b, x="Valor Líquido", y="Bairro", order=top10_b, palette="Blues_r", ax=ax1)
            ax1.set_xlabel("Aluguel Líquido (R$)")
            ax1.set_ylabel("Bairro")
            st.pyplot(fig1)

        with g_col2:
            st.markdown("**Impacto do Repasse Garantido na Retenção vs. Evasão**")
            if "Repasse Garantido" in df_imoveis.columns and "Status" in df_imoveis.columns:
                df_status = df_imoveis[df_imoveis["Status"].isin(["Alugado", "Baixado"])]
                ct_rep = pd.crosstab(df_status["Repasse Garantido"], df_status["Status"], normalize="index") * 100
                fig2, ax2 = plt.subplots(figsize=(8, 4.5))
                ct_rep.plot(kind="bar", stacked=False, color=["#2b83ba", "#d7191c"], edgecolor="black", ax=ax2)
                ax2.set_ylabel("Percentual da Carteira (%)")
                ax2.set_xlabel("Política de Repasse")
                ax2.tick_params(axis="x", rotation=15)
                st.pyplot(fig2)

        g_col3, g_col4 = st.columns(2)
        with g_col3:
            st.markdown("**Relação Área Total vs. Valor Líquido por Quantidade de Quartos**")
            df_sc = df_imoveis[(df_imoveis["Valor Líquido"] <= 20000) & (df_imoveis["Área Total (m2)"] <= 1000) & (df_imoveis["Área Total (m2)"] > 10)].copy()
            fig3, ax3 = plt.subplots(figsize=(8, 4.2))
            sns.scatterplot(data=df_sc, x="Área Total (m2)", y="Valor Líquido", hue="Status", alpha=0.6, ax=ax3)
            ax3.set_xlabel("Área Total (m²)")
            ax3.set_ylabel("Valor Líquido (R$)")
            st.pyplot(fig3)

        with g_col4:
            st.markdown("**Proporção da Carteira por Faixa do Proprietário**")
            if "Faixa_Carteira_Prop" in df_imoveis.columns:
                contagem_faixa = df_imoveis["Faixa_Carteira_Prop"].value_counts()
                fig4, ax4 = plt.subplots(figsize=(8, 4.2))
                ax4.pie(contagem_faixa.values, labels=contagem_faixa.index, autopct="%1.1f%%", startangle=140, colors=sns.color_palette("pastel"))
                st.pyplot(fig4)

# ================= ABA 3: MAPA GEOGRÁFICO INTERATIVO =================
with aba_mapa:
    st.subheader("🗺️ Localização Espacial dos Imóveis (Curitiba e Região)")
    if df_imoveis.empty or "Latitude" not in df_imoveis.columns:
        st.warning("Carregue a base de Imóveis contendo colunas de Latitude e Longitude.")
    else:
        status_disponiveis = list(df_imoveis["Status"].dropna().unique())
        filtro_status = st.multiselect("Filtrar por Status no Mapa:", status_disponiveis, default=status_disponiveis[:2])

        df_geo = df_imoveis[
            (df_imoveis["Latitude"] < -20.0) & (df_imoveis["Latitude"] > -30.0) &
            (df_imoveis["Longitude"] < -45.0) & (df_imoveis["Longitude"] > -55.0) &
            (df_imoveis["Status"].isin(filtro_status))
        ].copy()

        st.write(f"Exibindo **{len(df_geo)}** imóveis georreferenciados válidos (excluindo coordenadas zeradas).")
        df_geo_map = df_geo.rename(columns={"Latitude": "lat", "Longitude": "lon"})
        st.map(df_geo_map[["lat", "lon"]])

# ================= ABA 4: MACHINE LEARNING COM TREINAMENTO DINÂMICO =================
with aba_ml:
    st.subheader("🤖 Modelagem Preditiva Configurável e Retreinamento Dinâmico")
    if df_imoveis.empty:
        st.warning("Carregue a base de Imóveis na Aba 1 antes de iniciar o treinamento.")
    else:
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)

        with col_cfg1:
            objetivo = st.selectbox(
                "1. Escolha o Objetivo de Negócio:",
                options=[
                    ("evasao_proprietario", "Classificação: Risco de Evasão do Proprietário (Baixado vs Alugado)"),
                    ("quebra_contrato", "Classificação: Risco de Quebra Antecipada do Inquilino"),
                    ("preco_aluguel", "Regressão: Estimar Aluguel Líquido Ideal (R$)"),
                    ("meses_locados", "Regressão: Prever Tempo de Permanência (Meses)")
                ],
                format_func=lambda x: x[1]
            )[0]

        tipo_prob = "classificacao" if objetivo in ["evasao_proprietario", "quebra_contrato"] else "regressao"

        with col_cfg2:
            if tipo_prob == "classificacao":
                algoritmo = st.selectbox(
                    "2. Classificador Configurável:",
                    ["Árvore de Decisão", "KNN (Vizinhos Mais Próximos)", "Random Forest", "Regressão Logística"]
                )
            else:
                algoritmo = st.selectbox(
                    "2. Regressor Configurável:",
                    ["Regressão Linear (OLS)", "Regressão Ridge (Regularização L2)", "Árvore de Regressão", "Random Forest Regressor"]
                )

        with col_cfg3:
            usar_iqr = st.checkbox("Aplicar Cirurgia de Outliers (1.5 × IQR)", value=True)
            test_size = st.slider("Proporção do Conjunto de Teste (%)", 15, 40, 20) / 100.0

        # Painel de Hiperparâmetros Dinâmicos
        params = {"test_size": test_size}
        hp1, hp2 = st.columns(2)
        if "Árvore" in algoritmo:
            params["max_depth"] = hp1.slider("Profundidade Máxima da Árvore (max_depth):", 2, 20, 5)
            params["min_samples_split"] = hp2.slider("Mínimo de Amostras p/ Divisão:", 2, 30, 5)
        elif "KNN" in algoritmo:
            params["n_neighbors"] = hp1.slider("Número de Vizinhos (K):", 1, 25, 5)
            params["weights"] = hp2.selectbox("Peso da Distância:", ["uniform", "distance"])
        elif "Random Forest" in algoritmo:
            params["n_estimators"] = hp1.slider("Quantidade de Árvores (n_estimators):", 20, 200, 100, step=20)
            params["max_depth"] = hp2.slider("Profundidade Máxima (max_depth):", 2, 20, 8)
        elif "Ridge" in algoritmo:
            params["alpha"] = hp1.slider("Força da Regularização L2 (Alpha):", 0.1, 50.0, 1.0)
        elif "Logística" in algoritmo:
            params["C"] = hp1.slider("Inverso da Regularização (C):", 0.1, 10.0, 1.0)

        if st.button("🚀 Treinar / Re-treinar Modelo Agora com Base Atual", type="primary"):
            X, y, cols_num, cols_X = preparar_dataset_modelagem(df_imoveis, df_contratos, objetivo, usar_iqr)
            if X is None or len(X) < 15:
                st.error("Amostra insuficiente para este cruzamento. Faça o upload da base de Contratos Rescindidos na Aba 1.")
            else:
                res = treinar_modelo_dinamico(X, y, tipo_prob, algoritmo, params)
                st.session_state["ultimo_modelo"] = res
                st.session_state["ultimo_tipo"] = tipo_prob
                st.session_state["ultimas_features_num"] = cols_num
                st.session_state["X_mediana"] = X.median().to_dict()

        if "ultimo_modelo" in st.session_state:
            res = st.session_state["ultimo_modelo"]
            st.info(f"**Diagnóstico de Generalização (Aula 09/10):** {res['diagnostico']}")

            if st.session_state["ultimo_tipo"] == "classificacao":
                m1, m2, m3 = st.columns(3)
                m1.metric("Acurácia no Treino", f"{res['acc_train']*100:.1f}%")
                m2.metric("Acurácia no Teste (Inédito)", f"{res['acc_test']*100:.1f}%")
                m3.metric("F1-Score (Teste)", f"{res['f1_test']*100:.1f}%")

                fig_cm, ax_cm = plt.subplots(figsize=(5, 3.5))
                sns.heatmap(res["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax_cm)
                ax_cm.set_title("Matriz de Confusão no Conjunto de Teste")
                ax_cm.set_xlabel("Classe Prevista")
                ax_cm.set_ylabel("Classe Real")
                st.pyplot(fig_cm)
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R² Treino", f"{res['r2_train']*100:.1f}%")
                m2.metric("R² Teste", f"{res['r2_test']*100:.1f}%")
                m3.metric("MAE (Erro Médio Absoluto)", f"{res['mae']:,.2f}")
                m4.metric("RMSE (Erro Quadrático)", f"{res['rmse']:,.2f}")

            st.markdown("---")
            st.markdown("#### 🎯 Simulador Interativo de Predição Individual")
            valores_sim = st.session_state["X_mediana"].copy()
            cols_sim = st.columns(len(st.session_state["ultimas_features_num"]))
            for idx, f_name in enumerate(st.session_state["ultimas_features_num"]):
                valores_sim[f_name] = cols_sim[idx].number_input(
                    f"{f_name}", value=float(valores_sim.get(f_name, 0.0))
                )
            df_input = pd.DataFrame([valores_sim])[res["colunas_X"]]
            input_sc = res["scaler"].transform(df_input)
            pred_individual = res["modelo"].predict(input_sc)[0]

            if st.session_state["ultimo_tipo"] == "classificacao":
                status_pred = "ALTO RISCO (Classe Positiva = 1)" if pred_individual == 1 else "BAIXO RISCO / ESTÁVEL (Classe = 0)"
                st.success(f"**Resultado da Predição para o Cenário Informado:** {status_pred}")
            else:
                st.success(f"**Valor Contínuo Previsto pelo Modelo:** {pred_individual:,.2f}")

# ================= ABA 5: LOGS E AUDITORIA EM ARQUIVO TXT =================
with aba_logs:
    st.subheader("🛡️ Trilha de Auditoria e Persistência (audit_log.txt)")
    st.write("Registros gravados em disco via Context Manager (`with open(..., 'a', encoding='utf-8')`).")
    logs = repo.ler_historico_logs(40)
    st.code("\n".join(logs), language="text")