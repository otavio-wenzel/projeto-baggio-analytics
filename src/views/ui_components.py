"""
Camada View: Componentes globais de interface (CSS, Ampliação de Gráficos e Menu Lateral).
"""

import streamlit as st
import pandas as pd

from src.models.schemas import (
    CATALOGO_ANALISES_EDA,
    CATALOGO_SUBMODULOS_ML,
    CATALOGO_SUBMODULOS_CLUSTER
)
from src.models.repository import salvar_configuracoes
from src.controllers.data_processor import configurar_estilo_grafico


def exibir_grafico_responsivo(fig, chave_grafico: str):
    """Exibe o gráfico com opção de ampliar e fechar sem sobrepor seletores."""
    esta_ampliado = (st.session_state.grafico_ampliado_id == chave_grafico)

    if esta_ampliado:
        c_top1, c_top2 = st.columns([4, 1])
        c_top1.info("**Modo de Visualização Ampliada Ativo** — Clique no botão ao lado para fechar e retornar ao tamanho normal.")
        if c_top2.button("✖ Fechar Ampliação", key=f"fechar_top_{chave_grafico}", type="primary", use_container_width=True):
            st.session_state.grafico_ampliado_id = None
            st.rerun()

        st.pyplot(fig, use_container_width=True)

        if st.button("✖ Fechar Visualização Ampliada", key=f"fechar_bot_{chave_grafico}", use_container_width=True):
            st.session_state.grafico_ampliado_id = None
            st.rerun()
    else:
        col_esq, col_centro, col_dir = st.columns([1, 6, 1])
        with col_centro:
            st.pyplot(fig, use_container_width=True)
            if st.button("Ampliar Visualização do Gráfico", key=f"amp_{chave_grafico}"):
                st.session_state.grafico_ampliado_id = chave_grafico
                st.rerun()


def aplicar_css_global(tema: str):
    """Configura o contraste dos temas Claro e Escuro e oculta botões flutuantes intrusivos."""
    configurar_estilo_grafico(tema)

    css_comum = """
        .stDeployButton, [data-testid="stAppDeployButton"], [data-testid="stToolbarActions"], #MainMenu, footer {
            display: none !important;
        }
        header[data-testid="stHeader"] {
            background: transparent !important;
        }
        button[title="View fullscreen"], [data-testid="StyledFullScreenButton"] {
            display: none !important;
        }
        section[data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"] {
            display: none !important;
        }
    """

    if tema == "Claro":
        st.markdown(
            f"""
            <style>
            {css_comum}
            .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
                background-color: #f8fafc !important;
                color: #0f172a !important;
            }}
            section[data-testid="stSidebar"] {{
                background-color: #f1f5f9 !important;
                border-right: 1px solid #cbd5e1 !important;
            }}
            .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
            .stApp p, .stApp label, .stApp span, .stMarkdown,
            [data-testid="stWidgetLabel"] p, [data-testid="stMetricLabel"] p,
            [data-testid="stMetricValue"], [data-testid="stCaptionContainer"] p,
            section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
            section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] h3 {{
                color: #0f172a !important;
            }}
            div[data-testid="stMetric"] {{
                background-color: #ffffff !important;
                border: 1px solid #cbd5e1 !important;
                padding: 12px !important;
                border-radius: 10px !important;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
            }}
            div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
            input, textarea {{
                background-color: #ffffff !important;
                color: #0f172a !important;
                border-color: #94a3b8 !important;
            }}
            button[kind="secondary"] {{
                background-color: #ffffff !important;
                color: #0f172a !important;
                border: 1px solid #cbd5e1 !important;
            }}
            button[kind="secondary"] p, button[kind="secondary"] span {{
                color: #0f172a !important;
            }}
            button[kind="primary"] p, button[kind="primary"] span {{
                color: #ffffff !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    elif tema == "Escuro":
        st.markdown(
            f"""
            <style>
            {css_comum}
            .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
                background-color: #0f172a !important;
                color: #f8fafc !important;
            }}
            section[data-testid="stSidebar"] {{
                background-color: #1e293b !important;
                border-right: 1px solid #334155 !important;
            }}
            .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
            .stApp p, .stApp label, .stMarkdown,
            [data-testid="stWidgetLabel"] p, [data-testid="stMetricLabel"] p,
            [data-testid="stMetricValue"], [data-testid="stCaptionContainer"] p {{
                color: #f8fafc !important;
            }}
            div[data-testid="stMetric"] {{
                background-color: #1e293b !important;
                border: 1px solid #334155 !important;
                padding: 12px !important;
                border-radius: 10px !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(f"<style>{css_comum}</style>", unsafe_allow_html=True)


def renderizar_sidebar(config_usuario: dict, df_imoveis: pd.DataFrame, df_contratos: pd.DataFrame):
    """Renderiza o menu lateral com seletor de tema e submenus expansíveis."""
    with st.sidebar:
        st.markdown("### Baggio Intelligence")
        st.caption("Analytics, Governança LGPD & IA")

        opcoes_tema = ["Automático", "Claro", "Escuro"]
        idx_t = opcoes_tema.index(config_usuario.get("theme", "Automático")) if config_usuario.get("theme") in opcoes_tema else 0
        tema_sel = st.selectbox("Tema Visual", options=opcoes_tema, index=idx_t)
        if tema_sel != config_usuario.get("theme"):
            config_usuario["theme"] = tema_sel
            salvar_configuracoes(config_usuario)
            st.rerun()

        aplicar_css_global(tema_sel)

        st.divider()
        st.markdown("**Navegação do Sistema**")

        if st.button(
            "Visão Geral & Indicadores",
            use_container_width=True,
            type="primary" if st.session_state.rota_atual == "Visão Geral" else "secondary"
        ):
            st.session_state.rota_atual = "Visão Geral"
            st.rerun()

        if st.button(
            "Ingestão de Dados",
            use_container_width=True,
            type="primary" if st.session_state.rota_atual == "Ingestão de Dados" else "secondary"
        ):
            st.session_state.rota_atual = "Ingestão de Dados"
            st.rerun()

        with st.expander("Análises Exploratórias", expanded=(st.session_state.rota_atual == "Análises EDA")):
            for item_eda in CATALOGO_ANALISES_EDA:
                ativo_eda = (st.session_state.rota_atual == "Análises EDA" and st.session_state.analise_eda_ativa == item_eda)
                if st.button(
                    item_eda,
                    key=f"nav_eda_{item_eda}",
                    use_container_width=True,
                    type="primary" if ativo_eda else "secondary"
                ):
                    st.session_state.rota_atual = "Análises EDA"
                    st.session_state.analise_eda_ativa = item_eda
                    st.rerun()

        with st.expander("Laboratório de Predição (ML)", expanded=(st.session_state.rota_atual == "Laboratório ML")):
            for item_ml in CATALOGO_SUBMODULOS_ML:
                ativo_ml = (st.session_state.rota_atual == "Laboratório ML" and st.session_state.submodulo_ml_ativo == item_ml)
                if st.button(
                    item_ml,
                    key=f"nav_ml_{item_ml}",
                    use_container_width=True,
                    type="primary" if ativo_ml else "secondary"
                ):
                    st.session_state.rota_atual = "Laboratório ML"
                    st.session_state.submodulo_ml_ativo = item_ml
                    st.rerun()

        with st.expander("Clusterização & Anomalias (PCA)", expanded=(st.session_state.rota_atual == "Clusters & PCA")):
            for item_cl in CATALOGO_SUBMODULOS_CLUSTER:
                ativo_cl = (st.session_state.rota_atual == "Clusters & PCA" and st.session_state.submodulo_cluster_ativo == item_cl)
                if st.button(
                    item_cl,
                    key=f"nav_cl_{item_cl}",
                    use_container_width=True,
                    type="primary" if ativo_cl else "secondary"
                ):
                    st.session_state.rota_atual = "Clusters & PCA"
                    st.session_state.submodulo_cluster_ativo = item_cl
                    st.rerun()

        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Imóveis", len(df_imoveis))
        m2.metric("Contratos", len(df_contratos))