"""
Camada View: Tela de Ingestão de Dados em Lote, Cadastro Manual e CRUD Discreto.
"""

from datetime import date
import streamlit as st
import pandas as pd
import numpy as np

from src.models.schemas import (
    OPCOES_TIPO_IMOVEL, OPCOES_STATUS_IMOVEL, OPCOES_REPASSE,
    OPCOES_TX_INCIDE, OPCOES_MOTIVO_BAIXA, OPCOES_GARANTIA,
    OPCOES_TIPO_CONTRATO, OPCOES_MOTIVO_RESCISAO, BAIRROS_PRINCIPAIS
)
from src.models.repository import (
    ingerir_e_persistir_lote,
    inserir_registro_manual,
    atualizar_imovel_crud,
    remover_imovel_crud
)


def renderizar_ingestao_dados(df_imoveis: pd.DataFrame):
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
                    st.success(f"Imóvel `{ref_busca}` atualizado para **{novo_status}**!")
                    st.rerun()

                if b_del.form_submit_button("Excluir Imóvel"):
                    remover_imovel_crud(ref_busca)
                    st.warning(f"Imóvel `{ref_busca}` excluído.")
                    st.rerun()