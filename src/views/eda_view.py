import streamlit as st
import pandas as pd

from src.models.schemas import CATALOGO_ANALISES_EDA, formatar_rotulo
from src.views.ui_components import exibir_grafico_responsivo
from src.controllers.eda_temporal_ctrl import (
    plot_a1_escalada_anual,
    plot_a2_sazonalidade_mensal,
    plot_a3_ranking_motivos
)
from src.controllers.eda_politicas_ctrl import (
    plot_a4_repasse_garantido,
    plot_a5_taxa_administrativa,
    plot_a6_porte_carteira,
    plot_a7_geografia_bairros,
    plot_a8_garantias_iqr_corr,
    plot_a9_construtor_customizavel
)


def renderizar_modulo_analise(
    nome_analise: str,
    df_imoveis: pd.DataFrame,
    df_contratos: pd.DataFrame,
    sufixo_key: str = "main"
):
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


def renderizar_tela_eda(df_imoveis: pd.DataFrame, df_contratos: pd.DataFrame):
    st.subheader("Explorador de Análises Independentes & Compositor de Sessão")
    if df_imoveis.empty:
        st.info("Envie a Base de Imóveis na aba de Ingestão para liberar as visualizações.")
        return

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

    renderizar_modulo_analise(st.session_state.analise_eda_ativa, df_imoveis, df_contratos, sufixo_key="principal")

    for idx_ex, nome_extra in enumerate(analises_agrupadas):
        st.divider()
        renderizar_modulo_analise(nome_extra, df_imoveis, df_contratos, sufixo_key=f"extra_{idx_ex}")