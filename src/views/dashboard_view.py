import streamlit as st
import pandas as pd

from src.controllers.data_processor import preparar_base_cruzada


def renderizar_visao_geral(df_imoveis: pd.DataFrame, df_contratos: pd.DataFrame):
    st.subheader("Painel Executivo & Ferramentas da Plataforma")
    st.caption("Síntese operacional da carteira e acesso direto aos módulos analíticos e preditivos.")

    _, df_limpo, _ = preparar_base_cruzada(df_imoveis, df_contratos)
    taxa_evasao = (df_imoveis["Status"] == "Baixado").mean() * 100 if not df_imoveis.empty and "Status" in df_imoveis.columns else 0.0
    perm_media = df_limpo["Meses_Locados"].mean() if not df_limpo.empty else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Imóveis na Base", len(df_imoveis))
    k2.metric("Taxa Histórica de Baixa", f"{taxa_evasao:.1f}%")
    k3.metric("Contratos Saneados (IQR)", len(df_limpo))
    k4.metric("Permanência Média", f"{perm_media:.1f} meses")

    st.markdown("#### Acesso Rápido aos Módulos")
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Alimentar Base / CRUD", use_container_width=True):
        st.session_state.rota_atual = "Ingestão de Dados"
        st.rerun()
    if b2.button("Explorar as 9 Análises", use_container_width=True):
        st.session_state.rota_atual = "Análises EDA"
        st.rerun()
    if b3.button("Laboratório de Predição (ML)", use_container_width=True):
        st.session_state.rota_atual = "Laboratório ML"
        st.rerun()
    if b4.button("Mapear Personas & PCA", use_container_width=True):
        st.session_state.rota_atual = "Clusters & PCA"
        st.rerun()