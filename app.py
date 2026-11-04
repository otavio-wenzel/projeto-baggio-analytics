from datetime import date
import streamlit as st
import pandas as pd
import numpy as np

from src.schemas import (
    OPCOES_TIPO_IMOVEL, OPCOES_STATUS_IMOVEL, OPCOES_REPASSE,
    OPCOES_TX_INCIDE, OPCOES_MOTIVO_BAIXA, OPCOES_GARANTIA,
    OPCOES_TIPO_CONTRATO, OPCOES_MOTIVO_RESCISAO, BAIRROS_PRINCIPAIS,
    formatar_rotulo
)
from src.storage import (
    PATH_IMOVEIS, PATH_CONTRATOS,
    carregar_configuracoes, salvar_configuracoes,
    carregar_base_persistida, ingerir_e_persistir_lote,
    inserir_registro_manual, atualizar_imovel_crud, remover_imovel_crud
)
from src.analytics import (
    configurar_estilo_grafico, preparar_base_cruzada,
    plot_a1_escalada_anual, plot_a2_sazonalidade_mensal,
    plot_a3_ranking_motivos, plot_a4_repasse_garantido,
    plot_a5_taxa_administrativa, plot_a6_porte_carteira,
    plot_a7_geografia_bairros, plot_a8_garantias_iqr_corr,
    plot_a9_construtor_customizavel
)
from src.ml_engine import (
    treinar_e_comparar_regressores,
    treinar_classificador_configuravel,
    executar_pipeline_clusters
)

st.set_page_config(
    page_title="Baggio Intelligence | Analytics & ML",
    page_icon="B",
    layout="wide"
)

CATALOGO_ANALISES_EDA = [
    "Escalada Anual de Retiradas",
    "Sazonalidade Mensal (Picos Automáticos)",
    "Ranking de Motivos de Evasão / Rescisão",
    "Política Interna: Repasse Garantido",
    "Política Interna: Base da Taxa Adm.",
    "Perfil do Investidor (Porte da Carteira)",
    "Geografia de Evasão por Bairro",
    "Garantias Locatícias, IQR & Correlação",
    "Construtor Livre de Cruzamentos"
]

CATALOGO_SUBMODULOS_ML = [
    "Regressão: Permanência & Validação K-Fold",
    "Classificadores: Risco de Quebra & Evasão",
    "Simulador Executivo de Novos Contratos"
]

CATALOGO_SUBMODULOS_CLUSTER = [
    "Mapa PCA 2D & Segmentação K-Means",
    "Dendrograma Hierárquico de Ward",
    "Tabela de Personas & Auditoria DBSCAN"
]

config_usuario = carregar_configuracoes()

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


def exibir_grafico_responsivo(fig, chave_grafico: str):
    """
    Exibe o gráfico em tamanho proporcional e confortável na página e oferece
    um botão dedicado para Ampliar e um botão visível para Fechar a ampliação,
    sem nunca sobrepor seletores e sem depender da tecla ESC.
    """
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
    """
    Configura o contraste completo dos temas Claro e Escuro, oculta o botão Deploy,
    remove o botão nativo flutuante que sobrepunha os filtros dos gráficos e
    desativa qualquer legenda flutuante na barra lateral.
    """
    configurar_estilo_grafico(tema)

    css_comum = """
        .stDeployButton, [data-testid="stAppDeployButton"], [data-testid="stToolbarActions"], #MainMenu, footer {
            display: none !important;
        }
        header[data-testid="stHeader"] {
            background: transparent !important;
        }
        /* Oculta o botão flutuante nativo de tela cheia que ficava em cima dos filtros */
        button[title="View fullscreen"], [data-testid="StyledFullScreenButton"] {
            display: none !important;
        }
        /* Remove ícones de tooltip da barra lateral */
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


df_imoveis = carregar_base_persistida(PATH_IMOVEIS)
df_contratos = carregar_base_persistida(PATH_CONTRATOS)

# =====================================================================
# MENU LATERAL LIMPO (SEM LEGENDAS/TOOLTIPS E COM SUBMENUS EXPANSÍVEIS)
# =====================================================================
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

    # Submenu 1: 9 Análises Exploratórias Independentes
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

    # Submenu 2: Laboratório de Predição (ML) com Subcategorias
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

    # Submenu 3: Clusterização & Anomalias (PCA) com Subcategorias
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


# =====================================================================
# FUNÇÃO DE RENDERIZAÇÃO INDIVIDUAL DE CADA ANÁLISE EXPLORATÓRIA
# =====================================================================
def renderizar_modulo_analise(nome_analise: str, sufixo_key: str = "main"):
    st.markdown(f"#### {nome_analise}")

    match nome_analise:
        case "Escalada Anual de Retiradas":
            c1, c2, c3, c4 = st.columns(4)
            int_anos = c1.slider("Intervalo de Anos", 2020, 2026, (2022, 2026), key=f"a1_an_{sufixo_key}", help="Define o ano inicial e final da série.")
            formato_g = c2.selectbox("Tipo de Gráfico", ["Barras Verticais", "Linha com Marcadores", "Pizza"], key=f"a1_fm_{sufixo_key}", help="Altera a visualização do período.")
            metrica_g = c3.selectbox("Métrica Exibida", ["Quantidade Absoluta", "Participação (%)"], key=f"a1_mt_{sufixo_key}", help="Alterna entre contagem e percentual.")
            tipos_disp = df_imoveis["Tipo"].dropna().unique().tolist() if "Tipo" in df_imoveis.columns else []
            tipos_sel = c4.multiselect("Filtrar Tipos de Imóvel", options=tipos_disp, default=[], key=f"a1_tp_{sufixo_key}", help="Deixe vazio para todos os tipos.")
            fig = plot_a1_escalada_anual(df_imoveis, int_anos, tipos_sel, formato_g, metrica_g)
            exibir_grafico_responsivo(fig, f"fig_a1_{sufixo_key}")

        case "Sazonalidade Mensal (Picos Automáticos)":
            c1, c2, c3 = st.columns([2, 2, 3])
            int_anos_saz = c1.slider("Intervalo de Anos", 2020, 2026, (2022, 2026), key=f"a2_an_{sufixo_key}", help="Filtra os anos considerados na curva mensal.")
            int_meses_saz = c2.slider("Intervalo de Meses (1=Jan a 12=Dez)", 1, 12, (1, 12), key=f"a2_ms_{sufixo_key}", help="Recorta o período de meses analisado.")
            mot_disp = df_imoveis["Motivo de Baixa"].dropna().unique().tolist() if "Motivo de Baixa" in df_imoveis.columns else []
            padrao_m = [m for m in ["Uso Próprio | Desistência | Reforma", "Adm. Particular | Locou Direto", "Vendeu o imóvel"] if m in mot_disp]
            mot_sel = c3.multiselect("Motivos de Baixa Analisados", options=mot_disp, default=padrao_m, key=f"a2_mt_{sufixo_key}", help="Seleciona os motivos que compõem a análise sazonal.")

            op1, op2, op3 = st.columns([2, 2, 2])
            dest_picos = op1.checkbox("Detectar Picos Sazonais Automaticamente", value=True, key=f"a2_pc_{sufixo_key}", help="Identifica matematicamente períodos acima da média + desvio padrão.")
            sens_std = op2.slider("Sensibilidade do Detector (Desvios σ)", 0.1, 1.5, 0.5, step=0.1, key=f"a2_sd_{sufixo_key}", help="Valores menores detectam mais picos; valores maiores destacam apenas picos extremos.")
            most_num = op3.checkbox("Exibir Quantidades nos Pontos", value=True, key=f"a2_nm_{sufixo_key}", help="Mostra os números exatos sobre cada mês.")

            fig = plot_a2_sazonalidade_mensal(df_imoveis, int_anos_saz, int_meses_saz, mot_sel, dest_picos, sens_std, most_num)
            exibir_grafico_responsivo(fig, f"fig_a2_{sufixo_key}")

        case "Ranking de Motivos de Evasão / Rescisão":
            c1, c2, c3, c4 = st.columns(4)
            fonte_r = c1.selectbox("Base Analisada", ["Imóveis (Motivo de Baixa)", "Contratos (Motivo de Rescisão)"], key=f"a3_fn_{sufixo_key}", help="Alterna entre baixas de proprietários e rescisões de inquilinos.")
            int_anos_r = c2.slider("Intervalo de Anos", 2020, 2026, (2022, 2026), key=f"a3_an_{sufixo_key}", help="Filtra os anos avaliados no ranking.")
            top_n_r = c3.slider("Quantidade no Ranking", 3, 12, 8, key=f"a3_tp_{sufixo_key}", help="Número de motivos listados.")
            modo_pct = c4.selectbox("Unidade da Barra", ["Percentual (%)", "Contagem Absoluta"], key=f"a3_pc_{sufixo_key}", help="Exibe em percentual ou volume absoluto.")
            fig = plot_a3_ranking_motivos(df_imoveis, df_contratos, fonte_r, int_anos_r, top_n_r, (modo_pct == "Percentual (%)"))
            exibir_grafico_responsivo(fig, f"fig_a3_{sufixo_key}")

        case "Política Interna: Repasse Garantido":
            c1, c2, c3 = st.columns(3)
            faixa_alug = c1.slider("Faixa de Valor do Aluguel (R$)", 0.0, 25000.0, (0.0, 15000.0), step=250.0, key=f"a4_al_{sufixo_key}", help="Filtra imóveis pelo intervalo de valor líquido.")
            modo_ex = c2.selectbox("Métrica de Comparação", ["Percentual Relativo (%)", "Quantidade de Imóveis"], key=f"a4_md_{sufixo_key}", help="Compara percentual ou contagem de imóveis.")
            tipos_d4 = df_imoveis["Tipo"].dropna().unique().tolist() if "Tipo" in df_imoveis.columns else []
            tipos_s4 = c3.multiselect("Filtrar Tipo de Imóvel", options=tipos_d4, default=[], key=f"a4_tp_{sufixo_key}", help="Filtra categorias construtivas.")
            fig = plot_a4_repasse_garantido(df_imoveis, faixa_alug, tipos_s4, modo_ex)
            exibir_grafico_responsivo(fig, f"fig_a4_{sufixo_key}")

        case "Política Interna: Base da Taxa Adm.":
            c1, c2 = st.columns(2)
            min_imv = c1.slider("Mínimo de Imóveis por Regra", 5, 50, 15, key=f"a5_mn_{sufixo_key}", help="Oculta regras com poucos imóveis.")
            met_tx = c2.selectbox("Indicador Analisado", ["Taxa de Evasão (% Baixado)", "Aluguel Líquido Médio (R$)"], key=f"a5_mt_{sufixo_key}", help="Avalia taxa de perda ou valor médio do aluguel.")
            fig = plot_a5_taxa_administrativa(df_imoveis, min_imv, met_tx)
            exibir_grafico_responsivo(fig, f"fig_a5_{sufixo_key}")

        case "Perfil do Investidor (Porte da Carteira)":
            c1, c2 = st.columns(2)
            faixas_disp = df_imoveis["Faixa_Carteira_Prop"].dropna().unique().tolist() if "Faixa_Carteira_Prop" in df_imoveis.columns else []
            faixas_sel = c1.multiselect("Porte da Carteira do Proprietário", options=faixas_disp, default=faixas_disp, key=f"a6_fx_{sufixo_key}", help="Filtra as faixas de investidores.")
            met_cart = c2.selectbox("Métrica de Avaliação", ["Retenção vs. Evasão (%)", "Aluguel Líquido Médio (R$)", "Taxa de Condomínio Média (R$)"], key=f"a6_mt_{sufixo_key}", help="Escolhe a variável comparada por porte.")
            fig = plot_a6_porte_carteira(df_imoveis, faixas_sel, met_cart)
            exibir_grafico_responsivo(fig, f"fig_a6_{sufixo_key}")

        case "Geografia de Evasão por Bairro":
            c1, c2, c3 = st.columns(3)
            top_b = c1.slider("Quantidade de Bairros (Top Volume)", 5, 25, 12, key=f"a7_tp_{sufixo_key}", help="Número de bairros exibidos.")
            met_b = c2.selectbox("Indicador Geográfico", ["Taxa de Evasão (% Baixado)", "Mediana de Aluguel Líquido (R$)", "Mediana de Condomínio (R$)"], key=f"a7_mt_{sufixo_key}", help="Métrica avaliada em cada bairro.")
            ord_b = c3.selectbox("Ordenação", ["Maior para Menor", "Menor para Maior"], key=f"a7_or_{sufixo_key}", help="Ordem das barras horizontais.")
            fig = plot_a7_geografia_bairros(df_imoveis, top_b, met_b, ord_b)
            exibir_grafico_responsivo(fig, f"fig_a7_{sufixo_key}")

        case "Garantias Locatícias, IQR & Correlação":
            if df_contratos.empty:
                st.warning("Envie a base de Contratos Rescindidos para liberar esta análise.")
            else:
                c1, c2, c3 = st.columns(3)
                modo_v8 = c1.selectbox(
                    "Visualização Desejada",
                    ["Permanência Média por Garantia", "Auditoria Boxplot (Antes vs. Depois do IQR)", "Mapa de Calor (Correlação de Pearson)"],
                    key=f"a8_vs_{sufixo_key}",
                    help="Alterna entre o gráfico de garantias, o filtro de outliers e o mapa de calor."
                )
                mult_iqr8 = c2.slider("Multiplicador da Cerca IQR", 1.0, 3.0, 1.5, step=0.1, key=f"a8_iq_{sufixo_key}", help="Ajusta o corte de outliers temporais.")
                int_meses8 = c3.slider("Intervalo de Permanência (Meses)", 1.0, 150.0, (1.0, 120.0), step=2.0, key=f"a8_in_{sufixo_key}", help="Filtra contratos pela duração.")
                fig = plot_a8_garantias_iqr_corr(df_imoveis, df_contratos, mult_iqr8, int_meses8, modo_v8)
                exibir_grafico_responsivo(fig, f"fig_a8_{sufixo_key}")

        case _:
            cols_cat = [c for c in ["Status", "Tipo", "Bairro", "Repasse Garantido", "Tx. Adm. Incide Sobre", "Faixa_Carteira_Prop"] if c in df_imoveis.columns]
            cols_num = [c for c in ["Valor Líquido", "Taxa de Condominio", "Área Total (m2)", "Total de Quartos", "Qtd_Imoveis_Proprietario"] if c in df_imoveis.columns]
            c1, c2, c3, c4 = st.columns(4)
            x_c = c1.selectbox("Eixo X (Categoria)", options=cols_cat, index=0, format_func=formatar_rotulo, key=f"a9_x_{sufixo_key}", help="Variável categórica exibida com nome limpo.")
            y_c = c2.selectbox("Eixo Y (Métrica Numérica)", options=cols_num, index=0, format_func=formatar_rotulo, key=f"a9_y_{sufixo_key}", help="Variável numérica exibida com nome limpo.")
            t_c = c3.selectbox("Formato do Gráfico", ["Barras Verticais (Média)", "Barras Horizontais (Média)", "Boxplot (Quartis)", "Dispersão (Scatter)", "Pizza (Proporção de X)"], key=f"a9_t_{sufixo_key}")
            p_c = c4.slider("Filtro de Outliers (Percentil Máximo Y)", 85, 100, 95, key=f"a9_p_{sufixo_key}", help="Remove valores extremos acima do percentil escolhido.")
            fig = plot_a9_construtor_customizavel(df_imoveis, x_c, y_c, t_c, p_c)
            exibir_grafico_responsivo(fig, f"fig_a9_{sufixo_key}")


# =====================================================================
# FUNÇÃO DE RENDERIZAÇÃO INDIVIDUAL DOS SUBMÓDULOS DE MACHINE LEARNING
# =====================================================================
def renderizar_submodulo_ml(nome_submodulo: str, df_limpo: pd.DataFrame, sufixo_key: str = "ml_main"):
    st.markdown(f"#### {nome_submodulo}")

    ALGORITMOS_REG = [
        "Floresta Aleatória (Random Forest)",
        "Regressão Ridge (Penalidade L2)",
        "Regressão Lasso (Penalidade L1)",
        "Regressão Linear (OLS)",
        "Árvore de Decisão (Regressor)"
    ]
    ALGORITMOS_CLF = [
        "Floresta Aleatória (Classificador)",
        "Regressão Logística",
        "Vizinhos Mais Próximos (KNN)",
        "Árvore de Decisão (Classificador)"
    ]

    match nome_submodulo:
        case "Regressão: Permanência & Validação K-Fold":
            c1, c2 = st.columns([2, 2])
            alg_principal = c1.selectbox(
                "Modelo Preditivo Principal (Regressor):",
                options=ALGORITMOS_REG,
                key=f"ml_reg_alg_{sufixo_key}",
                help="Modelo usado para gerar o gráfico Real vs. Predito e calcular o impacto das variáveis."
            )
            opcoes_comp = [a for a in ALGORITMOS_REG if a != alg_principal]
            sugestao_padrao = [a for a in ["Regressão Ridge (Penalidade L2)", "Regressão Linear (OLS)", "Floresta Aleatória (Random Forest)"] if a in opcoes_comp][:2]

            algs_extras = c2.multiselect(
                "Comparar com Mais Modelos na Mesma Tabela (K-Fold):",
                options=opcoes_comp,
                default=sugestao_padrao,
                key=f"ml_comp_{alg_principal}_{sufixo_key}",
                help="Compara modelos adicionais na validação cruzada ao lado do Baseline."
            )

            p1, p2, p3, p4 = st.columns(4)
            alpha_ml = p1.slider("Regularização Alpha (Ridge/Lasso)", 0.1, 20.0, 2.0, key=f"ml_al_{sufixo_key}")
            depth_ml = p2.slider("Profundidade Máxima da Árvore", 2, 15, 4, key=f"ml_dp_{sufixo_key}")
            trees_ml = p3.slider("Quantidade de Árvores", 20, 300, 100, step=20, key=f"ml_tr_{sufixo_key}")
            split_ml = p4.slider("Proporção de Teste (%)", 15, 35, 20, key=f"ml_sp_{sufixo_key}") / 100.0

            _, _, df_comp, fig_diag = treinar_e_comparar_regressores(
                df_limpo, alg_principal, algs_extras, alpha_ml, depth_ml, trees_ml, split_ml
            )
            st.dataframe(df_comp, use_container_width=True)
            exibir_grafico_responsivo(fig_diag, f"fig_ml_reg_{sufixo_key}")

        case "Classificadores: Risco de Quebra & Evasão":
            cl1, cl2, cl3, cl4 = st.columns(4)
            alvo_risco = cl1.selectbox(
                "Evento de Risco Alvo",
                options=["Quebra_Antecipada", "Proprietario_Evadiu"],
                format_func=formatar_rotulo,
                key=f"ml_clf_alv_{sufixo_key}",
                help="Escolhe entre prever a quebra antecipada do contrato ou a evasão do proprietário."
            )
            alg_clf = cl2.selectbox(
                "Algoritmo Classificador",
                options=ALGORITMOS_CLF,
                key=f"ml_clf_alg_{sufixo_key}"
            )
            param_ck = cl3.slider("Parâmetro C (Logística) ou Vizinhos K (KNN)", 1.0, 15.0, 3.0, key=f"ml_clf_ck_{sufixo_key}")
            depth_clf = cl4.slider("Profundidade Máxima (Árvores)", 2, 15, 5, key=f"ml_clf_dp_{sufixo_key}")

            _, _, _, met_clf = treinar_classificador_configuravel(
                df_limpo, alvo_risco, alg_clf, param_ck, depth_clf, 100, 0.2
            )
            m1, m2 = st.columns(2)
            m1.metric("Acurácia no Conjunto de Teste", f"{met_clf['Acuracia_Pct']:.1f}%")
            m2.metric("F1-Score no Conjunto de Teste", f"{met_clf['F1_Score_Pct']:.1f}%")
            exibir_grafico_responsivo(met_clf["Figura_CM"], f"fig_ml_cm_{sufixo_key}")

        case _:
            c_m1, c_m2, c_m3 = st.columns(3)
            alg_sim_reg = c_m1.selectbox("Modelo de Previsão de Tempo", options=ALGORITMOS_REG, key=f"sim_reg_{sufixo_key}")
            alg_sim_clf = c_m2.selectbox("Modelo de Previsão de Risco", options=ALGORITMOS_CLF, key=f"sim_clf_{sufixo_key}")
            alvo_sim = c_m3.selectbox("Tipo de Risco Avaliado", options=["Quebra_Antecipada", "Proprietario_Evadiu"], format_func=formatar_rotulo, key=f"sim_alv_{sufixo_key}")

            modelo_reg, cols_X, _, _ = treinar_e_comparar_regressores(df_limpo, alg_sim_reg, [], 2.0, 4, 100, 0.2)
            clf_treinado, scaler_clf, _, met_clf = treinar_classificador_configuravel(df_limpo, alvo_sim, alg_sim_clf, 3.0, 5, 100, 0.2)

            s1, s2, s3, s4 = st.columns(4)
            in_alug = s1.number_input("Aluguel Líquido (R$)", 500.0, 25000.0, 2000.0, step=100.0, key=f"s_al_{sufixo_key}")
            in_cond = s2.number_input("Taxa de Condomínio (R$)", 0.0, 6000.0, 350.0, step=50.0, key=f"s_cd_{sufixo_key}")
            in_area = s3.number_input("Área Total (m²)", 15.0, 1000.0, 80.0, step=10.0, key=f"s_ar_{sufixo_key}")
            in_quar = s4.number_input("Total de Quartos", 0, 10, 2, key=f"s_qr_{sufixo_key}")

            s5, s6, s7, s8 = st.columns(4)
            in_gar = s5.selectbox("Modalidade de Garantia", list(OPCOES_GARANTIA), key=f"s_gr_{sufixo_key}")
            in_rep = s6.selectbox("Política de Repasse", list(OPCOES_REPASSE), key=f"s_rp_{sufixo_key}")
            in_bai = s7.selectbox("Bairro do Imóvel", list(BAIRROS_PRINCIPAIS), key=f"s_br_{sufixo_key}")
            in_cart = s8.slider("Imóveis do Proprietário na Carteira", 1, 40, 2, key=f"s_ct_{sufixo_key}")

            vetor = pd.DataFrame(0.0, index=[0], columns=cols_X)
            for c_al in ["Valor Líquido_contrato", "Valor Líquido"]:
                if c_al in vetor.columns:
                    vetor[c_al] = in_alug
            if "Taxa de Condominio" in vetor.columns:
                vetor["Taxa de Condominio"] = in_cond
            if "Área Total (m2)" in vetor.columns:
                vetor["Área Total (m2)"] = in_area
            if "Total de Quartos" in vetor.columns:
                vetor["Total de Quartos"] = in_quar
            if "Qtd_Imoveis_Proprietario" in vetor.columns:
                vetor["Qtd_Imoveis_Proprietario"] = in_cart
            if f"Garantia_{in_gar}" in vetor.columns:
                vetor[f"Garantia_{in_gar}"] = 1.0
            if f"Repasse Garantido_{in_rep}" in vetor.columns:
                vetor[f"Repasse Garantido_{in_rep}"] = 1.0
            if f"Bairro_Agrupado_{in_bai}" in vetor.columns:
                vetor[f"Bairro_Agrupado_{in_bai}"] = 1.0

            meses_previstos = max(1.0, float(modelo_reg.predict(vetor)[0]))
            media_carteira = float(df_limpo["Meses_Locados"].mean())
            delta_meses = meses_previstos - media_carteira

            vetor_scaled = scaler_clf.transform(vetor)
            prob_risco = (
                float(clf_treinado.predict_proba(vetor_scaled)[0][1] * 100)
                if hasattr(clf_treinado, "predict_proba")
                else float(clf_treinado.predict(vetor_scaled)[0] * 100)
            )

            r1, r2, r3 = st.columns(3)
            r1.metric(
                f"Permanência Estimada ({alg_sim_reg})",
                f"{meses_previstos:.1f} meses",
                delta=f"{delta_meses:+.1f} meses vs média da carteira"
            )
            r2.metric(
                f"Risco de {formatar_rotulo(alvo_sim)}",
                f"{prob_risco:.1f}%",
                delta="Risco Elevado" if prob_risco >= 50 else "Risco Controlado",
                delta_color="inverse"
            )
            r3.metric(
                "Acurácia / F1 do Classificador",
                f"{met_clf['Acuracia_Pct']:.1f}% / {met_clf['F1_Score_Pct']:.1f}%"
            )


# =====================================================================
# ROTEAMENTO PRINCIPAL DO SISTEMA
# =====================================================================
match st.session_state.rota_atual:

    case "Visão Geral":
        st.subheader("Painel Executivo & Ferramentas da Plataforma")
        st.caption("Síntese operacional da carteira e acesso direto aos módulos analíticos e preditivos.")

        df_valid, df_limpo, limite_iqr = preparar_base_cruzada(df_imoveis, df_contratos)
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
            st.session_state.rota_atual = "Ingestão & CRUD"
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

    case "Ingestão de Dados":
        st.subheader("Ingestão Inteligente de Arquivos & Governança LGPD")
        st.caption("Detecção automática da tabela por Conjuntos (Sets), Hashing SHA-256 em memória RAM e atualização sem duplicidades.")

        arquivos = st.file_uploader(
            "Envie planilhas ou arquivos estruturados (.xlsx, .csv, .json, .xml)",
            type=["xlsx", "xls", "csv", "json", "xml"],
            accept_multiple_files=True
        )

        if st.button("Validar Estrutura, Anonimizar (LGPD) e Salvar Lote", type="primary"):
            if not arquivos:
                st.warning("Selecione pelo menos um arquivo.")
            else:
                for arq in arquivos:
                    try:
                        r = ingerir_e_persistir_lote(arq)
                        st.success(f"`{arq.name}` → **{r['tipo']}** | Lidos: {r['lidos']} | Novos: {r['novos']} | Atualizados: {r['atualizados']} | Total: {r['total']}")
                    except Exception as e:
                        st.error(f"Falha em `{arq.name}`: {e}")
                st.rerun()

        st.markdown("#### Operações Manuais Discretas (Uso Pontual)")

        with st.expander("Inclusão Manual Avulsa (Imóvel ou Contrato Rescindido)", expanded=False):
            st.info(
                "**Aviso:** Priorize a ingestão via arquivos em lote. "
                "Nos cadastros manuais abaixo, todos os dados sensíveis informados (como CPF/CNPJ e Nome) "
                "recebem a mesma criptografia SHA-256 e descarte LGPD antes da gravação em disco."
            )
            tipo_cad = st.radio("Modelo de Registro:", ["Imóvel na Carteira", "Contrato Rescindido"], horizontal=True)

            if tipo_cad == "Imóvel na Carteira":
                with st.form("form_manual_imovel"):
                    f1, f2, f3, f4 = st.columns(4)
                    ref_i = f1.text_input("Referência do Imóvel *", value="IMV-9999")
                    bairro_i = f2.selectbox("Bairro *", options=list(BAIRROS_PRINCIPAIS) + ["Outro Bairro"])
                    tipo_i = f3.selectbox("Tipo *", options=list(OPCOES_TIPO_IMOVEL))
                    status_i = f4.selectbox("Status *", options=list(OPCOES_STATUS_IMOVEL))

                    f5, f6, f7, f8 = st.columns(4)
                    val_liq_i = f5.number_input("Valor Líquido (R$)", min_value=0.0, value=2200.0, step=100.0)
                    cond_i = f6.number_input("Condomínio (R$)", min_value=0.0, value=350.0, step=50.0)
                    area_i = f7.number_input("Área Total (m²)", min_value=10.0, value=85.0, step=5.0)
                    quartos_i = f8.number_input("Quartos", min_value=0, value=2, step=1)

                    f9, f10, f11, f12 = st.columns(4)
                    vagas_i = f9.number_input("Vagas de Garagem", min_value=0, value=1)
                    tx_adm_i = f10.number_input("Taxa Adm. (%)", min_value=0.0, max_value=30.0, value=10.0)
                    repasse_i = f11.selectbox("Repasse Garantido", options=list(OPCOES_REPASSE))
                    incide_i = f12.selectbox("Base da Taxa Administrativa", options=list(OPCOES_TX_INCIDE))

                    p1, p2, p3, p4 = st.columns(4)
                    cpf_prop = p1.text_input("CPF / CNPJ do Proprietário *", placeholder="000.000.000-00")
                    nome_prop = p2.text_input("Nome do Proprietário (Descartado LGPD)")
                    end_imovel = p3.text_input("Endereço (Descartado LGPD)")
                    meses_gar = p4.number_input("Meses Garantidos", min_value=0, value=3)

                    if st.form_submit_button("Criptografar PII e Persistir Imóvel"):
                        res_m = inserir_registro_manual("imovel", {
                            "Referência": ref_i.strip(), "Bairro": bairro_i, "Tipo": tipo_i, "Status": status_i,
                            "Valor Líquido": val_liq_i, "Taxa de Condominio": cond_i, "Área Total (m2)": area_i,
                            "Total de Quartos": quartos_i, "Vagas de Garagem": vagas_i, "Taxa Administrativa (%)": tx_adm_i,
                            "Repasse Garantido": repasse_i, "Tx. Adm. Incide Sobre": incide_i, "Meses Garantidos": meses_gar,
                            "Propr. - CPF | CPNJ": cpf_prop, "Propr. - Nome | Razão Social": nome_prop,
                            "Endereço": end_imovel, "Data de Cadastro": str(date.today())
                        })
                        st.success(f"Imóvel `{res_m['referencia']}` salvo! Hash SHA-256: `{res_m['hash_gerado']}`")
                        st.rerun()
            else:
                with st.form("form_manual_contrato"):
                    c1, c2, c3, c4 = st.columns(4)
                    ref_c = c1.text_input("Referência Contrato *", value="CTR-9999")
                    ref_imv_c = c2.text_input("Referência Imóvel *", value="IMV-9999")
                    tipo_c = c3.selectbox("Tipo de Contrato", options=list(OPCOES_TIPO_CONTRATO))
                    gar_c = c4.selectbox("Garantia", options=list(OPCOES_GARANTIA))

                    c5, c6, c7, c8 = st.columns(4)
                    dt_ini = c5.date_input("Data de Início", value=date(2024, 1, 10))
                    dt_ter = c6.date_input("Data de Término", value=date(2026, 7, 10))
                    dt_resc = c7.date_input("Data de Rescisão", value=date(2026, 4, 15))
                    val_c = c8.number_input("Valor Líquido (R$)", min_value=0.0, value=2100.0, step=100.0)

                    c9, c10, c11 = st.columns(3)
                    mot_resc = c9.selectbox("Motivo de Rescisão", options=list(OPCOES_MOTIVO_RESCISAO))
                    cpf_inq = c10.text_input("CPF/CNPJ Inquilino (Descartado LGPD)")
                    nome_inq = c11.text_input("Nome Inquilino (Descartado LGPD)")

                    if st.form_submit_button("Anonimizar PII e Persistir Contrato"):
                        res_mc = inserir_registro_manual("contrato", {
                            "Referência Contrato": ref_c.strip(), "Referência Imóvel": ref_imv_c.strip(),
                            "Tipo de Contrato": tipo_c, "Garantia": gar_c, "Data de Início": str(dt_ini),
                            "Data de Término": str(dt_ter), "Data de Rescisão": str(dt_resc),
                            "Valor Líquido": val_c, "Motivo de Rescisão": mot_resc, "Status": "Rescindido",
                            "Inq - CPF/CNPJ": cpf_inq, "Inq - Nome": nome_inq
                        })
                        st.success(f"Contrato `{res_mc['referencia']}` salvo com sucesso!")
                        st.rerun()

        with st.expander("Gestão de Ciclo de Vida do Imóvel (CRUD Discreto — Entrada, Baixa e Retorno)", expanded=False):
            st.info("Atualize o status de imóveis que sofreram baixa ou que retornaram para a carteira ativa da imobiliária.")
            if not df_imoveis.empty:
                lista_refs = df_imoveis["Referência"].astype(str).unique().tolist()
                ref_busca = st.selectbox("Selecionar Imóvel pela Referência:", options=lista_refs)
                linha_atual = df_imoveis[df_imoveis["Referência"].astype(str) == str(ref_busca)].iloc[0]

                with st.form("form_crud_imovel"):
                    u1, u2, u3, u4 = st.columns(4)
                    st_atual = str(linha_atual.get("Status", "Alugado"))
                    idx_st = OPCOES_STATUS_IMOVEL.index(st_atual) if st_atual in OPCOES_STATUS_IMOVEL else 0
                    novo_status = u1.selectbox("Status do Imóvel", options=list(OPCOES_STATUS_IMOVEL), index=idx_st)

                    mot_atual = str(linha_atual.get("Motivo de Baixa", "Não aplicável (Imóvel Ativo)"))
                    idx_mot = OPCOES_MOTIVO_BAIXA.index(mot_atual) if mot_atual in OPCOES_MOTIVO_BAIXA else 0
                    novo_motivo = u2.selectbox("Motivo de Baixa (Se Baixado)", options=list(OPCOES_MOTIVO_BAIXA), index=idx_mot)

                    novo_aluguel = u3.number_input("Valor Líquido (R$)", min_value=0.0, value=float(pd.to_numeric(linha_atual.get("Valor Líquido", 0), errors="coerce") or 0.0), step=100.0)
                    novo_cond = u4.number_input("Condomínio (R$)", min_value=0.0, value=float(pd.to_numeric(linha_atual.get("Taxa de Condominio", 0), errors="coerce") or 0.0), step=50.0)

                    u5, u6 = st.columns(2)
                    rep_atual = str(linha_atual.get("Repasse Garantido", OPCOES_REPASSE[0]))
                    idx_rep = OPCOES_REPASSE.index(rep_atual) if rep_atual in OPCOES_REPASSE else 0
                    novo_repasse = u5.selectbox("Política de Repasse", options=list(OPCOES_REPASSE), index=idx_rep)
                    novo_cpf_prop = u6.text_input("Novo CPF/CNPJ Proprietário (Opcional)")

                    b_up, b_del = st.columns(2)
                    if b_up.form_submit_button("Salvar Alterações / Reentrada"):
                        campos_up = {"Status": novo_status, "Valor Líquido": novo_aluguel, "Taxa de Condominio": novo_cond, "Repasse Garantido": novo_repasse}
                        if novo_status == "Baixado":
                            campos_up["Motivo de Baixa"] = novo_motivo
                            campos_up["Data de Baixa"] = str(date.today())
                        else:
                            campos_up["Motivo de Baixa"] = np.nan
                            campos_up["Data de Baixa"] = np.nan
                        if novo_cpf_prop.strip():
                            campos_up["Propr. - CPF | CPNJ"] = novo_cpf_prop.strip()
                        atualizar_imovel_crud(ref_busca, campos_up)
                        st.success(f"✅ Imóvel `{ref_busca}` atualizado para **{novo_status}**!")
                        st.rerun()

                    if b_del.form_submit_button("Excluir Imóvel"):
                        remover_imovel_crud(ref_busca)
                        st.warning(f"Imóvel `{ref_busca}` excluído.")
                        st.rerun()

    case "Análises EDA":
        st.subheader("Explorador de Análises Independentes & Compositor de Sessão")
        if df_imoveis.empty:
            st.info("Envie a Base de Imóveis na aba de Ingestão para liberar as visualizações.")
        else:
            c_pri, c_agr = st.columns([2, 2])
            idx_ativa = CATALOGO_ANALISES_EDA.index(st.session_state.analise_eda_ativa) if st.session_state.analise_eda_ativa in CATALOGO_ANALISES_EDA else 0
            escolha_principal = c_pri.selectbox(
                "Análise Independente Ativa:",
                options=CATALOGO_ANALISES_EDA,
                index=idx_ativa
            )
            if escolha_principal != st.session_state.analise_eda_ativa:
                st.session_state.analise_eda_ativa = escolha_principal
                st.rerun()

            opcoes_extras_eda = [a for a in CATALOGO_ANALISES_EDA if a != st.session_state.analise_eda_ativa]
            analises_agrupadas = c_agr.multiselect(
                "Agrupar Mais Análises nesta Mesma Sessão (Opcional):",
                options=opcoes_extras_eda,
                default=[],
                key=f"agr_eda_{st.session_state.analise_eda_ativa}"
            )

            renderizar_modulo_analise(st.session_state.analise_eda_ativa, sufixo_key="principal")

            for idx_ex, nome_extra in enumerate(analises_agrupadas):
                st.divider()
                renderizar_modulo_analise(nome_extra, sufixo_key=f"extra_{idx_ex}")

    case "Laboratório ML":
        st.subheader("Laboratório de Predição, Classificadores de Risco & Simulador")
        df_valid, df_limpo, limite_iqr = preparar_base_cruzada(df_imoveis, df_contratos)

        if len(df_limpo) < 15:
            st.info("Envie ambas as bases (Imóveis e Contratos) para liberar o treinamento dinâmico.")
        else:
            c_ml1, c_ml2 = st.columns([2, 2])
            idx_ml = CATALOGO_SUBMODULOS_ML.index(st.session_state.submodulo_ml_ativo) if st.session_state.submodulo_ml_ativo in CATALOGO_SUBMODULOS_ML else 0
            escolha_ml = c_ml1.selectbox(
                "Submódulo Principal do Laboratório:",
                options=CATALOGO_SUBMODULOS_ML,
                index=idx_ml
            )
            if escolha_ml != st.session_state.submodulo_ml_ativo:
                st.session_state.submodulo_ml_ativo = escolha_ml
                st.rerun()

            opcoes_extras_ml = [m for m in CATALOGO_SUBMODULOS_ML if m != st.session_state.submodulo_ml_ativo]
            ml_agrupados = c_ml2.multiselect(
                "Agrupar Mais Submódulos de ML nesta Sessão (Opcional):",
                options=opcoes_extras_ml,
                default=[],
                key=f"agr_ml_{st.session_state.submodulo_ml_ativo}"
            )

            renderizar_submodulo_ml(st.session_state.submodulo_ml_ativo, df_limpo, sufixo_key="ml_pri")

            for idx_m, mod_extra in enumerate(ml_agrupados):
                st.divider()
                renderizar_submodulo_ml(mod_extra, df_limpo, sufixo_key=f"ml_ex_{idx_m}")

    case _:
        st.subheader("Segmentação de Personas (K-Means & Ward) e Auditoria de Anomalias (DBSCAN + PCA)")
        _, df_limpo, _ = preparar_base_cruzada(df_imoveis, df_contratos)

        if len(df_limpo) < 15:
            st.info("Envie as bases de Imóveis e Contratos para liberar a clusterização.")
        else:
            c_cl1, c_cl2 = st.columns([2, 2])
            idx_cl = CATALOGO_SUBMODULOS_CLUSTER.index(st.session_state.submodulo_cluster_ativo) if st.session_state.submodulo_cluster_ativo in CATALOGO_SUBMODULOS_CLUSTER else 0
            escolha_cl = c_cl1.selectbox(
                "Visualização Principal de Clusterização:",
                options=CATALOGO_SUBMODULOS_CLUSTER,
                index=idx_cl
            )
            if escolha_cl != st.session_state.submodulo_cluster_ativo:
                st.session_state.submodulo_cluster_ativo = escolha_cl
                st.rerun()

            opcoes_extras_cl = [c for c in CATALOGO_SUBMODULOS_CLUSTER if c != st.session_state.submodulo_cluster_ativo]
            cl_agrupados = c_cl2.multiselect(
                "Agrupar Mais Vistas de Cluster nesta Sessão (Opcional):",
                options=opcoes_extras_cl,
                default=[],
                key=f"agr_cl_{st.session_state.submodulo_cluster_ativo}"
            )

            p1, p2, p3 = st.columns(3)
            k_cl = p1.slider("Número de Grupos (K-Means)", 2, 7, 4)
            eps_db = p2.slider("Raio de Vizinhança (DBSCAN)", 0.5, 2.5, 1.1, step=0.1)
            min_s = p3.slider("Mínimo de Pontos por Grupo", 2, 12, 5)

            _, resumo_p, anomalias_db, sil, fig_pca, fig_dendro = executar_pipeline_clusters(df_limpo, k_cl, eps_db, min_s)

            m1, m2 = st.columns(2)
            m1.metric("Coeficiente de Silhueta Médio S(i)", f"{sil:.3f}")
            m2.metric("Anomalias Isoladas pelo DBSCAN (-1)", len(anomalias_db))

            fila_cl = [st.session_state.submodulo_cluster_ativo] + cl_agrupados
            for idx_c, vista_c in enumerate(fila_cl):
                if idx_c > 0:
                    st.divider()
                st.markdown(f"#### {vista_c}")
                if vista_c == "Mapa PCA 2D & Segmentação K-Means":
                    exibir_grafico_responsivo(fig_pca, f"fig_pca_{idx_c}")
                elif vista_c == "🌳 2. Dendrograma Hierárquico de Ward":
                    exibir_grafico_responsivo(fig_dendro, f"fig_dendro_{idx_c}")
                else:
                    st.markdown("**Resumo Executivo das Personas (Médias por Grupo K-Means)**")
                    st.dataframe(resumo_p, use_container_width=True)
                    st.markdown(f"**Contratos Anômalos Auditados pelo DBSCAN ({len(anomalias_db)} registros)**")
                    cols_a = [c for c in ["Referência Imóvel", "Bairro", "Tipo", "Valor Líquido_contrato", "Taxa de Condominio", "Meses_Locados", "Qtd_Imoveis_Proprietario"] if c in anomalias_db.columns]
                    df_anom_limpo = anomalias_db[cols_a].copy()
                    df_anom_limpo.columns = [formatar_rotulo(c) for c in df_anom_limpo.columns]
                    st.dataframe(df_anom_limpo, use_container_width=True)