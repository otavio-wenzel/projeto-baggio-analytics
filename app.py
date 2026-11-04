from datetime import date
import streamlit as st
import pandas as pd
import numpy as np

from src.schemas import (
    OPCOES_TIPO_IMOVEL, OPCOES_STATUS_IMOVEL, OPCOES_REPASSE,
    OPCOES_TX_INCIDE, OPCOES_MOTIVO_BAIXA, OPCOES_GARANTIA,
    OPCOES_TIPO_CONTRATO, OPCOES_MOTIVO_RESCISAO, BAIRROS_PRINCIPAIS
)
from src.storage import (
    PATH_IMOVEIS, PATH_CONTRATOS,
    carregar_configuracoes, salvar_configuracoes,
    carregar_base_persistida, ingerir_e_persistir_lote,
    inserir_registro_manual, atualizar_imovel_crud, remover_imovel_crud
)
from src.analytics import (
    configurar_estilo_grafico, preparar_base_cruzada,
    plot_subgrupo_sazonalidade, plot_subgrupo_politicas,
    plot_subgrupo_geografia_carteira, plot_subgrupo_garantias_iqr,
    plot_explorador_customizavel
)
from src.ml_engine import (
    treinar_e_comparar_regressores,
    treinar_classificador_configuravel,
    executar_pipeline_clusters
)

st.set_page_config(
    page_title="Baggio Intelligence | Analytics & ML",
    page_icon="🏢",
    layout="wide"
)

# 1. Gestão de Tema (Claro, Escuro ou Automático) persistido em settings.json
config_usuario = carregar_configuracoes()

if "rota_atual" not in st.session_state:
    st.session_state.rota_atual = "🏠 Visão Geral & Atalhos"


def aplicar_css_tema(tema_escolhido: str):
    """Aplica estilização dinâmica para Modo Claro, Escuro ou Automático."""
    configurar_estilo_grafico(tema_escolhido)
    if tema_escolhido == "Claro":
        st.markdown(
            """
            <style>
            .stApp { background-color: #f8fafc; color: #0f172a; }
            section[data-testid="stSidebar"] { background-color: #f1f5f9; border-right: 1px solid #cbd5e1; }
            div[data-testid="stMetric"] { background-color: #ffffff; border: 1px solid #e2e8f0; padding: 12px; border-radius: 10px; }
            </style>
            """,
            unsafe_allow_html=True
        )
    elif tema_escolhido == "Escuro":
        st.markdown(
            """
            <style>
            .stApp { background-color: #0f172a; color: #f8fafc; }
            section[data-testid="stSidebar"] { background-color: #1e293b; border-right: 1px solid #334155; }
            div[data-testid="stMetric"] { background-color: #1e293b; border: 1px solid #334155; padding: 12px; border-radius: 10px; }
            </style>
            """,
            unsafe_allow_html=True
        )


# Carregamento das bases persistidas
df_imoveis = carregar_base_persistida(PATH_IMOVEIS)
df_contratos = carregar_base_persistida(PATH_CONTRATOS)

# =====================================================================
# MENU LATERAL COMPLETO: TEMA, NAVEGAÇÃO E METODOLOGIAS
# =====================================================================
with st.sidebar:
    st.markdown("### 🏢 Baggio Intelligence")
    st.caption("Plataforma de Dados & Machine Learning")

    opcoes_tema = ["Automático", "Claro", "Escuro"]
    idx_tema = opcoes_tema.index(config_usuario.get("theme", "Automático")) if config_usuario.get("theme") in opcoes_tema else 0
    tema_selecionado = st.selectbox(
        "🎨 Tema da Interface",
        options=opcoes_tema,
        index=idx_tema,
        help="Alterna entre o modo Claro, Escuro ou Automático e salva sua preferência em settings.json."
    )
    if tema_selecionado != config_usuario.get("theme"):
        config_usuario["theme"] = tema_selecionado
        salvar_configuracoes(config_usuario)
        st.rerun()

    aplicar_css_tema(tema_selecionado)

    st.divider()
    ROTAS_SISTEMA = [
        "🏠 Visão Geral & Atalhos",
        "📥 Ingestão, Cadastro & CRUD",
        "📊 Análises Exploratórias (EDA)",
        "🤖 Laboratório de Predição (ML)",
        "🎯 Clusterização & Anomalias (PCA)"
    ]
    rota_escolhida = st.radio(
        "🧭 Navegação do Sistema",
        options=ROTAS_SISTEMA,
        index=ROTAS_SISTEMA.index(st.session_state.rota_atual),
        help="Navegue entre os módulos operacionais, analíticos e preditivos da plataforma."
    )
    if rota_escolhida != st.session_state.rota_atual:
        st.session_state.rota_atual = rota_escolhida
        st.rerun()

    st.divider()
    c_m1, c_m2 = st.columns(2)
    c_m1.metric("Imóveis", len(df_imoveis), help="Total de imóveis únicos anonimizados e persistidos em disco.")
    c_m2.metric("Contratos", len(df_contratos), help="Total de contratos rescindidos únicos persistidos em disco.")

    with st.expander("🛠️ Metodologias Ativas", expanded=False):
        st.markdown(
            "- **Sets (`issubset`)**: Detecção estrutural\n"
            "- **SHA-256 + Drop**: Curadoria LGPD\n"
            "- **Escrita Atômica**: Persistência JSON\n"
            "- **1.5 × IQR**: Cirurgia de Outliers\n"
            "- **K-Fold + Baseline**: Regressão & Classificação\n"
            "- **K-Means, Ward, DBSCAN & PCA**: Clusters"
        )

# =====================================================================
# ROTEAMENTO PRINCIPAL VIA MATCH / CASE (AULA DE CONDICIONAIS)
# =====================================================================
match st.session_state.rota_atual:

    # -----------------------------------------------------------------
    # ROTA 1: VISÃO GERAL & ATALHOS RÁPIDOS
    # -----------------------------------------------------------------
    case "🏠 Visão Geral & Atalhos":
        st.subheader("Painel Executivo & Ferramentas do Sistema")
        st.caption("Resumo operacional da carteira e acesso rápido aos módulos de análise e inteligência artificial.")

        df_valid, df_limpo, limite_iqr = preparar_base_cruzada(df_imoveis, df_contratos)
        taxa_evasao = (
            (df_imoveis["Status"] == "Baixado").mean() * 100
            if not df_imoveis.empty and "Status" in df_imoveis.columns else 0.0
        )
        perm_media = df_limpo["Meses_Locados"].mean() if not df_limpo.empty else 0.0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Imóveis na Base", len(df_imoveis), help="Quantidade de imóveis únicos cadastrados no repositório local.")
        k2.metric("Taxa Histórica de Baixa", f"{taxa_evasao:.1f}%", help="Percentual de imóveis com status 'Baixado' na carteira.")
        k3.metric("Contratos Saneados (IQR)", len(df_limpo), help="Contratos cruzados que passaram pelo filtro de outliers 1.5 × IQR.")
        k4.metric("Permanência Média", f"{perm_media:.1f} meses", help="Tempo médio de locação dos contratos saneados.")

        st.markdown("#### ⚡ Acesso Rápido às Ferramentas")
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("📥 Alimentar Base / CRUD", use_container_width=True, help="Enviar planilhas (.xlsx, .csv, .json, .xml), cadastrar item avulso ou atualizar status de imóvel."):
            st.session_state.rota_atual = "📥 Ingestão, Cadastro & CRUD"
            st.rerun()
        if b2.button("📊 Explorar Gráficos & Subgrupos", use_container_width=True, help="Analisar sazonalidade, políticas internas, bairros e combinar múltiplos gráficos na tela."):
            st.session_state.rota_atual = "📊 Análises Exploratórias (EDA)"
            st.rerun()
        if b3.button("🤖 Simular Predições & Modelos", use_container_width=True, help="Treinar regressores e classificadores, comparar com Baseline e prever novos contratos."):
            st.session_state.rota_atual = "🤖 Laboratório de Predição (ML)"
            st.rerun()
        if b4.button("🎯 Mapear Personas & Anomalias", use_container_width=True, help="Executar K-Means, Dendrograma de Ward, DBSCAN (-1) e projeção PCA 2D."):
            st.session_state.rota_atual = "🎯 Clusterização & Anomalias (PCA)"
            st.rerun()

    # -----------------------------------------------------------------
    # ROTA 2: INGESTÃO EM LOTE, INCLUSÃO MANUAL E CRUD DE IMÓVEIS
    # -----------------------------------------------------------------
    case "📥 Ingestão, Cadastro & CRUD":
        st.subheader("Ingestão Inteligente de Arquivos & Governança LGPD")
        st.caption("Reconhecimento automático da estrutura por Conjuntos (Sets), anonimização SHA-256 em memória RAM e atualização sem duplicidades.")

        arquivos = st.file_uploader(
            "Envie planilhas ou arquivos estruturados (.xlsx, .csv, .json, .xml)",
            type=["xlsx", "xls", "csv", "json", "xml"],
            accept_multiple_files=True,
            help="Você pode enviar a base de Imóveis ou de Contratos Rescindidos com qualquer nome de arquivo; o sistema identifica o modelo pelas colunas."
        )

        if st.button("🚀 Validar Estrutura, Anonimizar (LGPD) e Salvar Lote", type="primary", help="Executa a leitura em RAM, criptografa CPFs/CNPJs, descarta dados sensíveis e faz o Upsert atômico."):
            if not arquivos:
                st.warning("Selecione ao menos um arquivo antes de iniciar o processamento.")
            else:
                for arq in arquivos:
                    try:
                        r = ingerir_e_persistir_lote(arq)
                        st.success(
                            f"✅ `{arq.name}` → **{r['tipo']}** | Lidos: {r['lidos']} | "
                            f"Novos: {r['novos']} | Atualizados: {r['atualizados']} | Total: {r['total']}"
                        )
                    except Exception as e:
                        st.error(f"❌ Falha em `{arq.name}`: {e}")
                st.rerun()

        st.markdown("#### 🔧 Operações Manuais Complementares (Uso Pontual)")

        # EXPANDER DISCRETO 1: INCLUSÃO MANUAL AVULSA COM CRIPTOGRAFIA LGPD
        with st.expander("➕ Inclusão Manual Avulsa (Imóvel ou Contrato Rescindido)", expanded=False):
            st.info(
                "💡 **Aviso de Boa Prática:** A alimentação principal do sistema deve ser realizada via upload de arquivos em lote. "
                "Utilize este formulário apenas para cadastros pontuais. Os dados sensíveis informados (como CPF/CNPJ e Nome) "
                "serão automaticamente criptografados em Hash SHA-256 e descartados antes de salvar em disco."
            )
            tipo_cad = st.radio(
                "Selecione o Modelo de Registro:",
                ["Imóvel na Carteira", "Contrato Rescindido"],
                horizontal=True,
                help="Define os campos do formulário conforme o esquema oficial de 115 colunas (Imóveis) ou 77 colunas (Contratos)."
            )

            if tipo_cad == "Imóvel na Carteira":
                with st.form("form_manual_imovel"):
                    f1, f2, f3, f4 = st.columns(4)
                    ref_i = f1.text_input("Referência do Imóvel *", value="IMV-9999", help="Chave primária única do imóvel.")
                    bairro_i = f2.selectbox("Bairro *", options=list(BAIRROS_PRINCIPAIS) + ["Outro Bairro"], help="Macrorregião geográfica mantida para análise.")
                    tipo_i = f3.selectbox("Tipo do Imóvel *", options=list(OPCOES_TIPO_IMOVEL), help="Categoria construtiva do imóvel.")
                    status_i = f4.selectbox("Status Inicial *", options=list(OPCOES_STATUS_IMOVEL), help="Situação operacional do imóvel na imobiliária.")

                    f5, f6, f7, f8 = st.columns(4)
                    val_liq_i = f5.number_input("Valor Líquido (R$)", min_value=0.0, value=2200.0, step=100.0, help="Valor líquido do aluguel.")
                    cond_i = f6.number_input("Taxa de Condomínio (R$)", min_value=0.0, value=350.0, step=50.0, help="Valor mensal do condomínio.")
                    area_i = f7.number_input("Área Total (m²)", min_value=10.0, value=85.0, step=5.0, help="Área total cadastrada.")
                    quartos_i = f8.number_input("Total de Quartos", min_value=0, value=2, step=1, help="Quantidade de dormitórios.")

                    f9, f10, f11, f12 = st.columns(4)
                    vagas_i = f9.number_input("Vagas de Garagem", min_value=0, value=1, step=1, help="Quantidade de vagas de estacionamento.")
                    tx_adm_i = f10.number_input("Taxa Administrativa (%)", min_value=0.0, max_value=30.0, value=10.0, step=0.5, help="Percentual cobrado pela administradora.")
                    repasse_i = f11.selectbox("Política de Repasse", options=list(OPCOES_REPASSE), help="Cláusula contratual de garantia de repasse.")
                    incide_i = f12.selectbox("Tx. Adm. Incide Sobre", options=list(OPCOES_TX_INCIDE), help="Base de incidência da taxa administrativa.")

                    st.markdown("**Dados do Proprietário e Localização (Serão Criptografados/Anonimizados via LGPD)**")
                    p1, p2, p3, p4 = st.columns(4)
                    cpf_prop = p1.text_input("CPF / CNPJ do Proprietário *", placeholder="000.000.000-00", help="Será convertido em Hash SHA-256 para cálculo do porte da carteira e depois descartado.")
                    nome_prop = p2.text_input("Nome / Razão Social", placeholder="Nome real (Descartado na gravação)", help="PII descartada em memória RAM.")
                    end_imovel = p3.text_input("Endereço / Rua", placeholder="Rua real (Descartada na gravação)", help="PII descartada em memória RAM.")
                    meses_gar = p4.number_input("Meses Garantidos", min_value=0, value=3, help="Quantidade de meses cobertos pelo repasse garantido.")

                    salvar_imv = st.form_submit_button("🔒 Criptografar PII e Persistir Imóvel", help="Salva o imóvel no banco aplicando as regras da LGPD.")
                    if salvar_imv:
                        dados_imv = {
                            "Referência": ref_i.strip(),
                            "Bairro": bairro_i,
                            "Tipo": tipo_i,
                            "Status": status_i,
                            "Valor Líquido": val_liq_i,
                            "Taxa de Condominio": cond_i,
                            "Área Total (m2)": area_i,
                            "Total de Quartos": quartos_i,
                            "Vagas de Garagem": vagas_i,
                            "Taxa Administrativa (%)": tx_adm_i,
                            "Repasse Garantido": repasse_i,
                            "Tx. Adm. Incide Sobre": incide_i,
                            "Meses Garantidos": meses_gar,
                            "Propr. - CPF | CPNJ": cpf_prop,
                            "Propr. - Nome | Razão Social": nome_prop,
                            "Endereço": end_imovel,
                            "Data de Cadastro": str(date.today())
                        }
                        res_m = inserir_registro_manual("imovel", dados_imv)
                        st.success(f"✅ Imóvel `{res_m['referencia']}` persistido! Hash SHA-256 gerado: `{res_m['hash_gerado']}` (PII descartada).")
                        st.rerun()

            else:
                with st.form("form_manual_contrato"):
                    c1, c2, c3, c4 = st.columns(4)
                    ref_c = c1.text_input("Referência do Contrato *", value="CTR-9999", help="Chave primária única do contrato.")
                    ref_imv_c = c2.text_input("Referência do Imóvel Vinculado *", value="IMV-9999", help="Chave estrangeira que cruza com a tabela de Imóveis.")
                    tipo_c = c3.selectbox("Tipo de Contrato", options=list(OPCOES_TIPO_CONTRATO), help="Modalidade residencial ou comercial.")
                    gar_c = c4.selectbox("Modalidade de Garantia", options=list(OPCOES_GARANTIA), help="Garantia locatícia associada ao contrato.")

                    c5, c6, c7, c8 = st.columns(4)
                    dt_ini = c5.date_input("Data de Início", value=date(2024, 1, 10), help="Data de início da vigência do contrato.")
                    dt_ter = c6.date_input("Data de Término Previsto", value=date(2026, 7, 10), help="Data original prevista para término.")
                    dt_resc = c7.date_input("Data de Rescisão Efetiva", value=date(2026, 4, 15), help="Data em que ocorreu a rescisão.")
                    val_c = c8.number_input("Valor Líquido (R$)", min_value=0.0, value=2100.0, step=100.0, help="Valor líquido pactuado no contrato.")

                    c9, c10, c11 = st.columns(3)
                    mot_resc = c9.selectbox("Motivo de Rescisão", options=list(OPCOES_MOTIVO_RESCISAO), help="Causa registrada para o encerramento.")
                    cpf_inq = c10.text_input("CPF/CNPJ do Inquilino", placeholder="000.000.000-00", help="Descartado automaticamente pela curadoria LGPD.")
                    nome_inq = c11.text_input("Nome do Inquilino", placeholder="Nome real", help="Descartado automaticamente pela curadoria LGPD.")

                    salvar_ctr = st.form_submit_button("🔒 Anonimizar PII e Persistir Contrato", help="Salva o contrato aplicando o descarte LGPD.")
                    if salvar_ctr:
                        dados_ctr = {
                            "Referência Contrato": ref_c.strip(),
                            "Referência Imóvel": ref_imv_c.strip(),
                            "Tipo de Contrato": tipo_c,
                            "Garantia": gar_c,
                            "Data de Início": str(dt_ini),
                            "Data de Término": str(dt_ter),
                            "Data de Rescisão": str(dt_resc),
                            "Valor Líquido": val_c,
                            "Motivo de Rescisão": mot_resc,
                            "Status": "Rescindido",
                            "Inq - CPF/CNPJ": cpf_inq,
                            "Inq - Nome": nome_inq
                        }
                        res_mc = inserir_registro_manual("contrato", dados_ctr)
                        st.success(f"✅ Contrato `{res_mc['referencia']}` persistido com proteção LGPD! Total de contratos: {res_mc['total']}.")
                        st.rerun()

        # EXPANDER DISCRETO 2: CRUD DE IMÓVEIS (ALTERAÇÃO DE STATUS, REENTRADA E EXCLUSÃO)
        with st.expander("🛠️ Gestão de Ciclo de Vida do Imóvel (CRUD Discreto — Entrada, Baixa e Retorno)", expanded=False):
            st.info(
                "💡 **Aviso Operacional:** Como um imóvel pode mudar de status ao longo do tempo (ser alugado, sofrer baixa "
                "e posteriormente retornar à carteira da imobiliária), utilize este módulo para atualizar seu status ou dados cadastrais."
            )
            if df_imoveis.empty:
                st.warning("Nenhum imóvel disponível na base para edição.")
            else:
                lista_refs = df_imoveis["Referência"].astype(str).unique().tolist()
                ref_busca = st.selectbox(
                    "🔎 Buscar Imóvel pela Referência:",
                    options=lista_refs,
                    help="Selecione o código de referência do imóvel que deseja inspecionar, alterar status ou excluir."
                )
                linha_atual = df_imoveis[df_imoveis["Referência"].astype(str) == str(ref_busca)].iloc[0]

                with st.form("form_crud_imovel"):
                    u1, u2, u3, u4 = st.columns(4)
                    st_atual = str(linha_atual.get("Status", "Alugado"))
                    idx_st = OPCOES_STATUS_IMOVEL.index(st_atual) if st_atual in OPCOES_STATUS_IMOVEL else 0
                    novo_status = u1.selectbox(
                        "Novo Status do Imóvel",
                        options=list(OPCOES_STATUS_IMOVEL),
                        index=idx_st,
                        help="Altere para 'Baixado' quando o imóvel sair da carteira, ou para 'Divulgado'/'Alugado' caso ele retorne."
                    )

                    mot_atual = str(linha_atual.get("Motivo de Baixa", "Não aplicável (Imóvel Ativo)"))
                    idx_mot = OPCOES_MOTIVO_BAIXA.index(mot_atual) if mot_atual in OPCOES_MOTIVO_BAIXA else 0
                    novo_motivo = u2.selectbox(
                        "Motivo de Baixa (Se Baixado)",
                        options=list(OPCOES_MOTIVO_BAIXA),
                        index=idx_mot,
                        help="Se o imóvel retornar à carteira como Ativo/Alugado/Divulgado, o sistema limpa a baixa automaticamente."
                    )

                    novo_aluguel = u3.number_input(
                        "Valor Líquido Atualizado (R$)",
                        min_value=0.0,
                        value=float(pd.to_numeric(linha_atual.get("Valor Líquido", 0), errors="coerce") or 0.0),
                        step=100.0,
                        help="Atualiza o valor líquido do aluguel."
                    )
                    novo_cond = u4.number_input(
                        "Condomínio Atualizado (R$)",
                        min_value=0.0,
                        value=float(pd.to_numeric(linha_atual.get("Taxa de Condominio", 0), errors="coerce") or 0.0),
                        step=50.0,
                        help="Atualiza o valor da taxa de condomínio."
                    )

                    u5, u6 = st.columns(2)
                    rep_atual = str(linha_atual.get("Repasse Garantido", OPCOES_REPASSE[0]))
                    idx_rep = OPCOES_REPASSE.index(rep_atual) if rep_atual in OPCOES_REPASSE else 0
                    novo_repasse = u5.selectbox("Política de Repasse Garantido", options=list(OPCOES_REPASSE), index=idx_rep, help="Atualiza a cláusula de garantia de repasse.")
                    novo_cpf_prop = u6.text_input("Novo CPF/CNPJ (Opcional — caso mude o proprietário)", placeholder="Deixe em branco para manter o Hash atual", help="Se preenchido, gera novo Hash SHA-256 e recalcula a carteira.")

                    b_up, b_del = st.columns(2)
                    btn_atualizar = b_up.form_submit_button("💾 Salvar Alterações / Reentrada", help="Grava o novo status e valores com escrita atômica.")
                    btn_excluir = b_del.form_submit_button("🗑️ Excluir Imóvel da Base", help="Remove o registro da base persistida.")

                    if btn_atualizar:
                        campos_up = {
                            "Status": novo_status,
                            "Valor Líquido": novo_aluguel,
                            "Taxa de Condominio": novo_cond,
                            "Repasse Garantido": novo_repasse
                        }
                        if novo_status == "Baixado":
                            campos_up["Motivo de Baixa"] = novo_motivo
                            campos_up["Data de Baixa"] = str(date.today())
                        else:
                            # Imóvel retornou para a imobiliária: limpa os campos de baixa
                            campos_up["Motivo de Baixa"] = np.nan
                            campos_up["Data de Baixa"] = np.nan

                        if novo_cpf_prop.strip():
                            campos_up["Propr. - CPF | CPNJ"] = novo_cpf_prop.strip()

                        atualizar_imovel_crud(ref_busca, campos_up)
                        st.success(f"✅ Imóvel `{ref_busca}` atualizado para Status **{novo_status}**!")
                        st.rerun()

                    if btn_excluir:
                        remover_imovel_crud(ref_busca)
                        st.warning(f"🗑️ Imóvel `{ref_busca}` removido da base.")
                        st.rerun()

    # -----------------------------------------------------------------
    # ROTA 3: ANÁLISES EXPLORATÓRIAS CUSTOMIZÁVEIS & COMPOSITOR
    # -----------------------------------------------------------------
    case "📊 Análises Exploratórias (EDA)":
        st.subheader("Explorador Analítico por Subgrupos & Compositor Visual")
        if df_imoveis.empty:
            st.info("Envie a Base de Imóveis no módulo de Ingestão para habilitar os gráficos.")
        else:
            SUBGRUPOS_EDA = [
                "1. Evolução & Sazonalidade de Retiradas",
                "2. Políticas Internas (Repasse & Taxa Adm.)",
                "3. Geografia (Bairros) & Porte do Investidor",
                "4. Garantias Locatícias & Saneamento IQR",
                "5. Construtor Livre de Cruzamentos (Customizável)"
            ]

            c_sel, c_comp = st.columns([2, 2])
            subgrupo_ativo = c_sel.selectbox(
                "📂 Escolha o Subgrupo de Análise Principal:",
                options=SUBGRUPOS_EDA,
                help="Exibe apenas o grupo analítico selecionado com controles de customização dedicados."
            )
            vistas_extras = c_comp.multiselect(
                "🧩 Agrupar Mais Análises nesta Mesma Sessão (Opcional):",
                options=[s for s in SUBGRUPOS_EDA if s != subgrupo_ativo],
                help="Permite combinar 2 ou mais subgrupos analíticos na mesma página para comparação integrada."
            )

            fila_renderizacao = [subgrupo_ativo] + vistas_extras

            for vista in fila_renderizacao:
                st.markdown(f"#### {vista}")
                match vista:
                    case "1. Evolução & Sazonalidade de Retiradas":
                        p1, p2, p3 = st.columns(3)
                        ano_min = p1.slider("Ano Inicial da Série", 2018, 2026, 2022, key=f"ano_{vista}", help="Filtra os imóveis baixados a partir do ano selecionado.")
                        tipo_g1 = p2.selectbox("Formato do Gráfico Anual", ["Barras", "Linha", "Pizza"], key=f"tg_{vista}", help="Alterna entre gráfico de Barras, Linha ou Pizza (com autopct).")
                        motivos_disp = df_imoveis["Motivo de Baixa"].dropna().unique().tolist() if "Motivo de Baixa" in df_imoveis.columns else []
                        padrao_mot = [m for m in ["Uso Próprio | Desistência | Reforma", "Adm. Particular | Locou Direto", "Vendeu o imóvel"] if m in motivos_disp]
                        mot_sel = p3.multiselect("Motivos na Curva Sazonal", options=motivos_disp, default=padrao_mot, key=f"mot_{vista}", help="Seleciona quais motivos de baixa aparecem no gráfico de linha mensal.")
                        st.pyplot(plot_subgrupo_sazonalidade(df_imoveis, ano_min, mot_sel, tipo_g1))

                    case "2. Políticas Internas (Repasse & Taxa Adm.)":
                        p1, p2 = st.columns(2)
                        var_pol = p1.radio("Variável de Política Interna:", ["Repasse Garantido", "Tx. Adm. Incide Sobre"], horizontal=True, key=f"vp_{vista}", help="Alterna o cruzamento entre a cláusula de Repasse e a regra da Taxa Administrativa.")
                        min_am = p2.slider("Mínimo de Imóveis por Categoria", 5, 50, 10, key=f"ma_{vista}", help="Oculta categorias residuais com poucos imóveis.")
                        st.pyplot(plot_subgrupo_politicas(df_imoveis, var_pol, min_am))

                    case "3. Geografia (Bairros) & Porte do Investidor":
                        top_n = st.slider("Quantidade de Bairros no Ranking", 5, 20, 12, key=f"tn_{vista}", help="Define quantos bairros com maior volume serão comparados.")
                        st.pyplot(plot_subgrupo_geografia_carteira(df_imoveis, top_n))

                    case "4. Garantias Locatícias & Saneamento IQR":
                        if df_contratos.empty:
                            st.warning("Envie a base de Contratos Rescindidos para visualizar garantias e outliers.")
                        else:
                            mult_iqr = st.slider("Multiplicador da Cerca IQR (Padrão = 1.5)", 1.0, 3.0, 1.5, step=0.1, key=f"iqr_{vista}", help="Ajusta o rigor do corte de outliers temporais (Q3 + k × IQR).")
                            df_v, df_l, lim = preparar_base_cruzada(df_imoveis, df_contratos, mult_iqr)
                            st.pyplot(plot_subgrupo_garantias_iqr(df_v, df_l, lim))

                    case _:
                        cols_cat = [c for c in ["Status", "Tipo", "Bairro", "Repasse Garantido", "Tx. Adm. Incide Sobre", "Faixa_Carteira_Prop"] if c in df_imoveis.columns]
                        cols_num = [c for c in ["Valor Líquido", "Taxa de Condominio", "Área Total (m2)", "Total de Quartos", "Qtd_Imoveis_Proprietario"] if c in df_imoveis.columns]
                        e1, e2, e3 = st.columns(3)
                        x_c = e1.selectbox("Variável Categórica / Eixo X", options=cols_cat, index=0, key=f"xc_{vista}", help="Define o agrupamento no eixo horizontal.")
                        y_c = e2.selectbox("Variável Numérica / Eixo Y", options=cols_num, index=0, key=f"yc_{vista}", help="Define a métrica numérica analisada.")
                        t_c = e3.selectbox("Tipo de Visualização", ["Barras (Média)", "Boxplot", "Dispersão (Scatter)", "Pizza (Contagem de X)"], key=f"tc_{vista}", help="Escolhe o estilo do gráfico gerado.")
                        st.pyplot(plot_explorador_customizavel(df_imoveis, x_c, y_c, t_c))

    # -----------------------------------------------------------------
    # ROTA 4: LABORATÓRIO DE PREDIÇÃO & TREINAMENTO DINÂMICO
    # -----------------------------------------------------------------
    case "🤖 Laboratório de Predição (ML)":
        st.subheader("Laboratório de Machine Learning, Validação Cruzada & Simulador")
        df_valid, df_limpo, limite_iqr = preparar_base_cruzada(df_imoveis, df_contratos)

        if len(df_limpo) < 15:
            st.info("Envie ambas as bases (Imóveis e Contratos) para liberar o treinamento dinâmico dos modelos.")
        else:
            ALGORITMOS_REG = [
                "RandomForestRegressor",
                "Ridge (Penalidade L2)",
                "Lasso (Penalidade L1)",
                "LinearRegression (OLS)",
                "DecisionTreeRegressor"
            ]
            c1, c2 = st.columns([2, 2])
            alg_principal = c1.selectbox(
                "🎯 Modelo Preditivo Principal (Regressor):",
                options=ALGORITMOS_REG,
                help="Algoritmo principal usado no simulador de predição e no gráfico Real vs. Predito."
            )
            algs_extras = c2.multiselect(
                "📊 Comparar com Mais Modelos na Mesma Sessão (K-Fold):",
                options=[a for a in ALGORITMOS_REG if a != alg_principal],
                default=["Ridge (Penalidade L2)", "LinearRegression (OLS)"] if alg_principal != "Ridge (Penalidade L2)" else ["RandomForestRegressor"],
                help="Inclui outros modelos na tabela de Validação Cruzada K-Fold ao lado do Baseline (Dummy)."
            )

            p1, p2, p3, p4 = st.columns(4)
            alpha_ml = p1.slider("Regularização Alpha (Ridge/Lasso)", 0.1, 20.0, 2.0, help="Controla a penalidade L1/L2 para evitar Overfitting.")
            depth_ml = p2.slider("Profundidade Máxima (max_depth)", 2, 15, 4, help="Limita a profundidade das árvores para evitar memorização do treino.")
            trees_ml = p3.slider("Nº de Árvores (n_estimators)", 20, 300, 100, step=20, help="Quantidade de árvores no comitê do Random Forest.")
            split_ml = p4.slider("Fração de Teste (%)", 15, 35, 20, help="Percentual separado exclusivamente para avaliação final.") / 100.0

            modelo_reg, cols_X, df_comp, fig_diag = treinar_e_comparar_regressores(
                df_limpo, alg_principal, algs_extras, alpha_ml, depth_ml, trees_ml, split_ml
            )

            st.markdown("##### 📋 Comparativo com Baseline (`DummyRegressor`) e Validação Cruzada (K-Fold)")
            st.dataframe(df_comp, use_container_width=True)
            st.pyplot(fig_diag)

            # Treinamento simultâneo do Classificador de Risco configurável
            st.divider()
            st.markdown("#### 🔮 Simulador Executivo Dual: Previsão de Permanência + Risco de Quebra/Evasão")
            cl1, cl2, cl3 = st.columns(3)
            alvo_risco = cl1.selectbox("Alvo do Classificador de Risco", ["Quebra_Antecipada", "Proprietario_Evadiu"], help="Define se o classificador prevê quebra de contrato pelo inquilino ou baixa pelo proprietário.")
            alg_clf = cl2.selectbox("Algoritmo Classificador", ["RandomForestClassifier", "LogisticRegression", "KNN (K-Nearest Neighbors)", "DecisionTreeClassifier"], help="Classificador configurável pelo usuário.")
            param_ck = cl3.slider("Hiperparâmetro C (LogReg) / Vizinhos K (KNN)", 1.0, 15.0, 3.0, help="Ajusta a regularização ou número de vizinhos do classificador.")

            clf_treinado, scaler_clf, _, met_clf = treinar_classificador_configuravel(
                df_limpo, alvo_risco, alg_clf, param_ck, depth_ml, trees_ml, split_ml
            )

            s1, s2, s3, s4 = st.columns(4)
            in_alug = s1.number_input("Aluguel Líquido (R$)", 500.0, 25000.0, 2000.0, step=100.0, help="Valor mensal do aluguel simulado.")
            in_cond = s2.number_input("Condomínio (R$)", 0.0, 6000.0, 350.0, step=50.0, help="Taxa de condomínio do imóvel simulado.")
            in_area = s3.number_input("Área Total (m²)", 15.0, 1000.0, 80.0, step=10.0, help="Metragem total do imóvel.")
            in_quar = s4.number_input("Quartos", 0, 10, 2, help="Total de quartos do imóvel.")

            s5, s6, s7, s8 = st.columns(4)
            in_gar = s5.selectbox("Garantia Locatícia", list(OPCOES_GARANTIA), help="Modalidade de garantia escolhida para o contrato.")
            in_rep = s6.selectbox("Política de Repasse", list(OPCOES_REPASSE), help="Regra de repasse garantido ao proprietário.")
            in_bai = s7.selectbox("Bairro do Imóvel", list(BAIRROS_PRINCIPAIS), help="Localização geográfica do imóvel.")
            in_cart = s8.slider("Imóveis do Proprietário", 1, 40, 2, help="Quantidade de imóveis que este proprietário possui na Baggio.")

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
                f"Permanência Prevista ({alg_principal})",
                f"{meses_previstos:.1f} meses",
                delta=f"{delta_meses:+.1f}m vs média da carteira",
                help="Estimativa gerada pelo modelo de regressão selecionado."
            )
            r2.metric(
                f"Probabilidade de {alvo_risco} ({alg_clf})",
                f"{prob_risco:.1f}%",
                delta="Alto Risco" if prob_risco >= 50 else "Risco Controlado",
                delta_color="inverse",
                help="Probabilidade calculada pelo classificador supervisionado."
            )
            r3.metric(
                f"Acurácia / F1 do Classificador",
                f"{met_clf['Acuracia_Pct']:.1f}% / {met_clf['F1_Score_Pct']:.1f}%",
                help="Desempenho do classificador no conjunto de teste."
            )

    # -----------------------------------------------------------------
    # ROTA 5: CLUSTERIZAÇÃO, PERSONAS & ANOMALIAS (PCA / DBSCAN)
    # -----------------------------------------------------------------
    case _:
        st.subheader("Segmentação de Personas (K-Means & Ward) e Auditoria de Anomalias (DBSCAN + PCA)")
        _, df_limpo, _ = preparar_base_cruzada(df_imoveis, df_contratos)

        if len(df_limpo) < 15:
            st.info("Envie as bases de Imóveis e Contratos para liberar a clusterização multidimensional.")
        else:
            p1, p2, p3 = st.columns(3)
            k_cl = p1.slider("Número de Clusters (K-Means)", 2, 7, 4, help="Define a quantidade de grupos (Personas) na segmentação K-Means.")
            eps_db = p2.slider("Raio de Vizinhança (eps DBSCAN)", 0.5, 2.5, 1.1, step=0.1, help="Define a distância máxima de densidade antes de rotular um contrato como anomalia (-1).")
            min_s = p3.slider("Mínimo de Pontos (min_samples)", 2, 12, 5, help="Densidade mínima exigida para formar um cluster válido no DBSCAN.")

            _, resumo_p, anomalias_db, sil, fig_pca, fig_dendro = executar_pipeline_clusters(
                df_limpo, k_cl, eps_db, min_s
            )

            m1, m2 = st.columns(2)
            m1.metric("Coeficiente de Silhueta S(i)", f"{sil:.3f}", help="Avalia a coesão interna e separação entre os clusters K-Means.")
            m2.metric("Anomalias Isoladas pelo DBSCAN (-1)", len(anomalias_db), help="Contratos atípicos detectados na periferia multidimensional.")

            vistas_cl = st.multiselect(
                "🧩 Escolha as Visualizações de Cluster para Exibir na Sessão:",
                options=["Mapa PCA 2D (K-Means vs DBSCAN)", "Dendrograma Hierárquico de Ward", "Tabela de Personas & Auditoria de Anomalias (-1)"],
                default=["Mapa PCA 2D (K-Means vs DBSCAN)", "Tabela de Personas & Auditoria de Anomalias (-1)"],
                help="Permite agrupar ou isolar os gráficos de aprendizado não supervisionado."
            )

            if "Mapa PCA 2D (K-Means vs DBSCAN)" in vistas_cl:
                st.pyplot(fig_pca)
            if "Dendrograma Hierárquico de Ward" in vistas_cl:
                st.pyplot(fig_dendro)
            if "Tabela de Personas & Auditoria de Anomalias (-1)" in vistas_cl:
                st.markdown("**Resumo das Personas (Médias por Cluster K-Means)**")
                st.dataframe(resumo_p, use_container_width=True)
                st.markdown(f"**🎯 Contratos Anômalos Auditados pelo DBSCAN (`Cluster == -1`: {len(anomalias_db)} registros)**")
                cols_a = [c for c in ["Referência Imóvel", "Bairro", "Tipo", "Valor Líquido_contrato", "Taxa de Condominio", "Meses_Locados", "Qtd_Imoveis_Proprietario"] if c in anomalias_db.columns]
                st.dataframe(anomalias_db[cols_a], use_container_width=True)