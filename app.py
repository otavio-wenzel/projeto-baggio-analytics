import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.storage import (
    PATH_IMOVEIS, PATH_CONTRATOS,
    carregar_base_persistida, ingerir_e_persistir_lote
)
from src.analytics import (
    preparar_base_cruzada,
    plot_escalada_e_sazonalidade,
    plot_fatores_internos_e_carteira,
    plot_outliers_e_heatmap
)
from src.ml_engine import (
    treinar_regressor_configuravel,
    treinar_classificador_configuravel,
    executar_pipeline_clusters
)

st.set_page_config(
    page_title="Baggio Intelligence | Analytics & ML",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Plataforma de Inteligência de Dados & Machine Learning — Imobiliária Baggio")
st.markdown(
    "Sistema web com **reconhecimento estrutural automático**, **curadoria LGPD em RAM**, "
    "**persistência atômica não-volátil** e **retreinamento dinâmico de modelos**."
)

# Carregamento do estado persistido em disco
df_imoveis = carregar_base_persistida(PATH_IMOVEIS)
df_contratos = carregar_base_persistida(PATH_CONTRATOS)

# Barra lateral de status do Banco de Dados Local (Flat Files JSON)
with st.sidebar:
    st.header("🗄️ Estado da Persistência (Disco)")
    st.metric("Imóveis Persistidos (Únicos)", len(df_imoveis))
    st.metric("Contratos Rescindidos (Únicos)", len(df_contratos))
    st.caption("Armazenamento seguro via Escrita Atômica (UTF-8) e deduplicado por Chave Primária.")

tab_ingestao, tab_eda, tab_ml_sup, tab_clusters = st.tabs([
    "📥 1. Ingestão Inteligente & LGPD",
    "📊 2. Diagnóstico Visual (EDA)",
    "🤖 3. Predições & Classificadores (ML)",
    "🎯 4. Clusterização & Anomalias (PCA)"
])

# =====================================================================
# ABA 1: UPLOAD FLEXÍVEL, RECONHECIMENTO ESTRUTURAL E PERSISTÊNCIA LGPD
# =====================================================================
with tab_ingestao:
    st.subheader("Upload Flexível com Detecção Automática de Estrutura")
    st.write(
        "Envie arquivos nos formatos **.csv, .json, .xml ou .xlsx**. O sistema ignora o título do arquivo, "
        "inspeciona o cabeçalho via **Conjuntos (Sets)** para identificar se é a base de **Imóveis** ou de "
        "**Contratos Rescindidos**, aplica o **Hashing SHA-256 (LGPD)** em memória RAM e realiza o **Upsert** sem duplicidades."
    )

    arquivos_enviados = st.file_uploader(
        "Selecione um ou mais arquivos de dados (.csv, .json, .xml, .xlsx)",
        type=["csv", "json", "xml", "xlsx", "xls"],
        accept_multiple_files=True
    )

    if st.button("🚀 Processar, Anonimizar (LGPD) e Persistir Dados", type="primary"):
        if not arquivos_enviados:
            st.warning("Selecione pelo menos um arquivo antes de processar.")
        else:
            for arq in arquivos_enviados:
                try:
                    res = ingerir_e_persistir_lote(arq)
                    st.success(
                        f"✅ Arquivo `{arq.name}` reconhecido como **{res['tipo']}**! "
                        f"Lidos: {res['lidos_no_arquivo']} | Novos absorvidos: {res['novos_absorvidos']} | "
                        f"Atualizados sem duplicar: {res['duplicados_atualizados']} | "
                        f"Total no banco: {res['total_persistido']}."
                    )
                except Exception as erro:
                    st.error(f"❌ Erro ao validar `{arq.name}`: {erro}")
            st.rerun()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Amostra da Base de Imóveis Persistida (Curada LGPD)**")
        if not df_imoveis.empty:
            cols_exibir_i = [
                c for c in ["Referência", "Bairro", "Tipo", "Status", "Valor Líquido",
                            "Hash_Proprietario", "Qtd_Imoveis_Proprietario", "Faixa_Carteira_Prop"]
                if c in df_imoveis.columns
            ]
            st.dataframe(df_imoveis[cols_exibir_i].head(8), use_container_width=True)
        else:
            st.info("Nenhuma base de imóveis persistida ainda. Faça o upload acima.")

    with col_b:
        st.markdown("**Amostra da Base de Contratos Persistida (Curada LGPD)**")
        if not df_contratos.empty:
            cols_exibir_c = [
                c for c in ["Referência Contrato", "Referência Imóvel", "Data de Início",
                            "Data de Rescisão", "Garantia", "Valor Líquido"]
                if c in df_contratos.columns
            ]
            st.dataframe(df_contratos[cols_exibir_c].head(8), use_container_width=True)
        else:
            st.info("Nenhuma base de contratos persistida ainda. Faça o upload acima.")

# =====================================================================
# ABA 2: ANÁLISE DE DADOS E VISUALIZAÇÃO (EDA)
# =====================================================================
with tab_eda:
    if df_imoveis.empty:
        st.info("Envie a Base de Imóveis na Aba 1 para liberar os diagnósticos visuais.")
    else:
        st.subheader("Diagnóstico da Carteira de Imóveis e Retenção de Proprietários")
        st.pyplot(plot_escalada_e_sazonalidade(df_imoveis))
        st.pyplot(plot_fatores_internos_e_carteira(df_imoveis))

        if not df_contratos.empty:
            st.subheader("Cruzamento Relacional (Merge) e Saneamento de Outliers (1.5 × IQR)")
            df_valid, df_limpo, limite_iqr = preparar_base_cruzada(df_imoveis, df_contratos)
            if not df_limpo.empty:
                c1, c2, c3 = st.columns(3)
                c1.metric("Contratos Cruzados Brutos", len(df_valid))
                c2.metric("Limite Superior (1.5 × IQR)", f"{limite_iqr:.1f} meses")
                c3.metric("Contratos Saneados p/ IA", len(df_limpo))
                st.pyplot(plot_outliers_e_heatmap(df_valid, df_limpo, limite_iqr))

# =====================================================================
# ABA 3: MACHINE LEARNING CONFIGURÁVEL & SIMULADOR DE PREDIÇÕES
# =====================================================================
with tab_ml_sup:
    df_valid, df_limpo, limite_iqr = preparar_base_cruzada(df_imoveis, df_contratos)
    if len(df_limpo) < 15:
        st.info("Envie ambas as bases (Imóveis e Contratos Rescindidos) na Aba 1 para habilitar o treinamento dinâmico de Machine Learning.")
    else:
        st.subheader("Configuração de Modelos Supervisionados e Retreinamento Dinâmico")
        modo_ml = st.radio(
            "Escolha a Tarefa de Machine Learning:",
            ["📈 Regressão: Prever Tempo de Permanência (Meses Locados)",
             "🛡️ Classificador: Prever Risco de Quebra Antecipada ou Evasão"],
            horizontal=True
        )

        if "Regressão" in modo_ml:
            c_alg, c_param, c_split = st.columns(3)
            alg_reg = c_alg.selectbox(
                "Algoritmo Regressor",
                ["LinearRegression (OLS)", "Ridge (Penalidade L2)", "Lasso (Penalidade L1)", "RandomForestRegressor"]
            )
            alpha_val = c_param.slider("Hiperparâmetro Alpha (Ridge/Lasso)", 0.01, 20.0, 1.0)
            n_est_reg = c_param.slider("Nº de Árvores (Random Forest)", 20, 300, 100)
            split_reg = c_split.slider("Proporção de Teste (%)", 15, 40, 20) / 100.0

            modelo_reg, cols_modelo, metricas_reg, df_pesos = treinar_regressor_configuravel(
                df_limpo, alg_reg, alpha_val, n_est_reg, split_reg
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Erro Médio Absoluto (MAE)", f"{metricas_reg['MAE']:.2f} meses")
            m2.metric("Raiz do Erro Quadrático (RMSE)", f"{metricas_reg['RMSE']:.2f} meses")
            m3.metric("Coeficiente R² (Teste)", f"{metricas_reg['R2_Pct']:.2f}%")
            m4.metric("Base de Treino / Teste", f"{metricas_reg['Amostras_Treino']} / {metricas_reg['Amostras_Teste']}")

            # Gráfico dos Pesos Beta / Importâncias
            top_p = pd.concat([df_pesos.head(5), df_pesos.tail(5)]).drop_duplicates()
            fig_b, ax_b = plt.subplots(figsize=(10, 4.2))
            cores = ["#d7191c" if v < 0 else "#1a9641" for v in top_p["Peso"]]
            ax_b.barh(top_p["Feature"], top_p["Peso"], color=cores, edgecolor="black")
            ax_b.axvline(0, color="black", lw=1.2)
            ax_b.set_title(f"Impacto das Variáveis no Modelo ({alg_reg})", fontweight="bold")
            st.pyplot(fig_b)

            # Simulador Interativo de Predição Personalizada
            st.markdown("#### 🔮 Simulador de Predição Personalizada (Novo Contrato)")
            s1, s2, s3, s4 = st.columns(4)
            sim_aluguel = s1.number_input("Aluguel Líquido (R$)", 500.0, 20000.0, 2000.0, step=100.0)
            sim_cond = s2.number_input("Taxa de Condomínio (R$)", 0.0, 5000.0, 350.0, step=50.0)
            sim_area = s3.number_input("Área Total (m²)", 20.0, 1000.0, 80.0, step=10.0)
            sim_quartos = s4.number_input("Total de Quartos", 0, 10, 2)

            s5, s6, s7 = st.columns(3)
            sim_garantia = s5.selectbox("Modalidade de Garantia", ["Seguro Fiança", "Fiador", "Título de Capitalização", "Sem Garantia"])
            sim_repasse = s6.selectbox("Política de Repasse", ["Garantir por alguns meses", "Não garante repasse"])
            sim_carteira = s7.slider("Imóveis do Proprietário na Baggio", 1, 50, 1)

            vetor_sim = pd.DataFrame(0, index=[0], columns=cols_modelo, dtype=float)
            for col_a in ["Valor Líquido_contrato", "Valor Líquido"]:
                if col_a in vetor_sim.columns:
                    vetor_sim[col_a] = sim_aluguel
            if "Taxa de Condominio" in vetor_sim.columns:
                vetor_sim["Taxa de Condominio"] = sim_cond
            if "Área Total (m2)" in vetor_sim.columns:
                vetor_sim["Área Total (m2)"] = sim_area
            if "Total de Quartos" in vetor_sim.columns:
                vetor_sim["Total de Quartos"] = sim_quartos
            if "Qtd_Imoveis_Proprietario" in vetor_sim.columns:
                vetor_sim["Qtd_Imoveis_Proprietario"] = sim_carteira
            if f"Garantia_{sim_garantia}" in vetor_sim.columns:
                vetor_sim[f"Garantia_{sim_garantia}"] = 1.0
            if f"Repasse Garantido_{sim_repasse}" in vetor_sim.columns:
                vetor_sim[f"Repasse Garantido_{sim_repasse}"] = 1.0

            pred_meses = float(modelo_reg.predict(vetor_sim)[0])
            st.success(f"⏱️ **Permanência Estimada pelo Modelo ({alg_reg}):** `{max(1.0, pred_meses):.1f} meses`")

        else:
            c1, c2, c3, c4 = st.columns(4)
            alvo_clf = c1.selectbox("Variável Alvo (Target)", ["Quebra_Antecipada", "Proprietario_Evadiu"])
            alg_clf = c2.selectbox("Classificador", ["RandomForestClassifier", "LogisticRegression", "KNN (K-Nearest Neighbors)"])
            param_ck = c3.slider("Parâmetro C (LogReg) ou K Vizinhos (KNN)", 1.0, 15.0, 3.0)
            n_est_clf = c4.slider("Nº de Árvores (Random Forest)", 20, 300, 100)

            _, _, _, met_clf = treinar_classificador_configuravel(
                df_limpo, alvo_clf, alg_clf, param_ck, n_est_clf
            )
            k1, k2 = st.columns(2)
            k1.metric("Acurácia no Conjunto de Teste", f"{met_clf['Acuracia_Pct']:.2f}%")
            k2.metric("F1-Score no Conjunto de Teste", f"{met_clf['F1_Score_Pct']:.2f}%")
            st.write("**Matriz de Confusão (Teste):**", met_clf["Matriz_Confusao"])

# =====================================================================
# ABA 4: CLUSTERIZAÇÃO K-MEANS, HIERÁRQUICO DE WARD, DBSCAN E PCA 2D
# =====================================================================
with tab_clusters:
    _, df_limpo, _ = preparar_base_cruzada(df_imoveis, df_contratos)
    if len(df_limpo) < 15:
        st.info("Envie as bases na Aba 1 para liberar a Clusterização e Caça a Anomalias.")
    else:
        st.subheader("Segmentação de Personas (K-Means & Ward) e Auditoria de Anomalias (DBSCAN + PCA)")
        p1, p2, p3 = st.columns(3)
        k_escolhido = p1.slider("Número de Clusters (K-Means)", 2, 7, 4)
        eps_val = p2.slider("Raio de Densidade (eps do DBSCAN)", 0.5, 2.5, 1.1, step=0.1)
        min_samp = p3.slider("Amostras Mínimas (min_samples DBSCAN)", 2, 12, 5)

        df_clust, resumo_p, anomalias_db, sil_score, fig_pca, fig_dendro = executar_pipeline_clusters(
            df_limpo, k_escolhido, eps_val, min_samp
        )

        st.metric("Coeficiente de Silhueta Médio S(i)", f"{sil_score:.3f}")
        st.pyplot(fig_pca)
        st.pyplot(fig_dendro)

        st.markdown("**Resumo Executivo dos Clusters (Personas K-Means)**")
        st.dataframe(resumo_p, use_container_width=True)

        st.markdown(f"**🎯 Anomalias Operacionais Detectadas pelo DBSCAN (Rótulo -1: {len(anomalias_db)} contratos)**")
        cols_aud = [
            c for c in ["Referência Imóvel", "Bairro", "Tipo", "Valor Líquido_contrato",
                        "Taxa de Condominio", "Meses_Locados", "Qtd_Imoveis_Proprietario"]
            if c in anomalias_db.columns
        ]
        st.dataframe(anomalias_db[cols_aud], use_container_width=True)