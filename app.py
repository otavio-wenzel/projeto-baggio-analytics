import streamlit as st

from src.models.schemas import (
    CATALOGO_ANALISES_EDA,
    CATALOGO_SUBMODULOS_ML,
    CATALOGO_SUBMODULOS_CLUSTER
)
from src.models.repository import (
    PATH_IMOVEIS,
    PATH_CONTRATOS,
    carregar_configuracoes,
    carregar_base_persistida
)
from src.views.ui_components import renderizar_sidebar
from src.views.dashboard_view import renderizar_visao_geral
from src.views.ingestao_view import renderizar_ingestao_dados
from src.views.eda_view import renderizar_tela_eda
from src.views.ml_view import renderizar_tela_ml
from src.views.clusters_view import renderizar_tela_clusters

st.set_page_config(
    page_title="Baggio Intelligence | Analytics & ML",
    page_icon="B",
    layout="wide"
)

if "rota_atual" not in st.session_state:
    st.session_state.rota_atual = "Visão Geral"
if "analise_eda_ativa" not in st.session_state:
    st.session_state.analise_eda_ativa = CATALOGO_ANALISES_EDA[0]
if "submodulo_ml_ativo" not in st.session_state:
    st.session_state.submodulo_ml_ativo = CATALOGO_SUBMODULOS_ML[0]
if "submodulo_cluster_ativo" not in st.session_state:
    st.session_state.submodulo_cluster_ativo = CATALOGO_SUBMODULOS_CLUSTER[0]
if "grafico_ampliado_id" not in st.session_state:
    st.session_state.grafico_ampliado_id = None

config_usuario = carregar_configuracoes()
df_imoveis = carregar_base_persistida(PATH_IMOVEIS)
df_contratos = carregar_base_persistida(PATH_CONTRATOS)

renderizar_sidebar(config_usuario, df_imoveis, df_contratos)

match st.session_state.rota_atual:
    case "Visão Geral":
        renderizar_visao_geral(df_imoveis, df_contratos)
    case "Ingestão de Dados":
        renderizar_ingestao_dados(df_imoveis)
    case "Análises EDA":
        renderizar_tela_eda(df_imoveis, df_contratos)
    case "Laboratório ML":
        renderizar_tela_ml(df_imoveis, df_contratos)
    case _:
        renderizar_tela_clusters(df_imoveis, df_contratos)