"""
Camada View: Tela do Laboratório de Predição (ML), Classificadores e Simulador Executivo.
"""

import streamlit as st
import pandas as pd

from src.models.schemas import (
    CATALOGO_SUBMODULOS_ML, OPCOES_GARANTIA, OPCOES_REPASSE,
    BAIRROS_PRINCIPAIS, formatar_rotulo
)
from src.views.ui_components import exibir_grafico_responsivo
from src.controllers.data_processor import preparar_base_cruzada
from src.controllers.ml_supervisionado_ctrl import (
    treinar_e_comparar_regressores,
    treinar_classificador_configuravel
)


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


def renderizar_tela_ml(df_imoveis: pd.DataFrame, df_contratos: pd.DataFrame):
    st.subheader("Laboratório de Predição, Classificadores de Risco & Simulador")
    _, df_limpo, _ = preparar_base_cruzada(df_imoveis, df_contratos)

    if len(df_limpo) < 15:
        st.info("Envie ambas as bases (Imóveis e Contratos) para liberar o treinamento dinâmico.")
        return

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