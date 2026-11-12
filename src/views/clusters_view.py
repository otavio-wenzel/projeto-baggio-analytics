import streamlit as st
import pandas as pd

from src.models.schemas import CATALOGO_SUBMODULOS_CLUSTER, formatar_rotulo
from src.views.ui_components import exibir_grafico_responsivo
from src.controllers.data_processor import preparar_base_cruzada
from src.controllers.ml_clusters_ctrl import executar_pipeline_clusters

def renderizar_tela_clusters(df_imoveis: pd.DataFrame, df_contratos: pd.DataFrame):
    st.subheader("Segmentação de Personas (K-Means & Ward) e Auditoria de Anomalias (DBSCAN + PCA)")
    
    _, df_limpo, _ = preparar_base_cruzada(df_imoveis, df_contratos)

    if len(df_limpo) < 15:
        st.info("Envie as bases de Imóveis e Contratos para liberar a clusterização.")
        return

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
        label="Agrupar Mais Vistas de Cluster nesta Sessão (Opcional):",
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
        elif vista_c == "Dendrograma Hierárquico de Ward":
            exibir_grafico_responsivo(fig_dendro, f"fig_dendro_{idx_c}")
        else:
            st.markdown("**Resumo Executivo das Personas (Médias por Grupo K-Means)**")
            st.dataframe(resumo_p, use_container_width=True)
            
            st.markdown(f"**Contratos Anômalos Auditados pelo DBSCAN ({len(anomalias_db)} registros)**")
            
            cols_a = [c for c in ["Referência Imóvel", "Bairro", "Tipo", "Valor Líquido_contrato", "Taxa de Condominio", "Meses_Locados", "Qtd_Imoveis_Proprietario"] if c in anomalias_db.columns]
            df_anom_limpo = anomalias_db[cols_a].copy()
            df_anom_limpo.columns = [formatar_rotulo(c) for c in df_anom_limpo.columns]
            st.dataframe(df_anom_limpo, use_container_width=True)