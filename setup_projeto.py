import json
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd

from database import carregar_configuracoes, registrar_log_auditoria
from data_processor import higienizar_imoveis, higienizar_contratos


def construir_estrutura_pastas():
    """Cria as pastas do projeto utilizando pathlib.Path (Aula 05)."""
    pastas = [
        Path("data_store"),
        Path("samples_upload"),
        Path("raw_private")
    ]
    for p in pastas:
        p.mkdir(parents=True, exist_ok=True)
        print(f"📁 Diretório verificado/criado: {p.resolve()}")

    carregar_configuracoes()
    registrar_log_auditoria("Ambiente inicializado via setup_projeto.py")
    print("⚙️ Arquivo settings.json inicializado com sucesso.")


def exportar_contratos_para_xml(df: pd.DataFrame, destino_xml: Path):
    """
    Exporta um DataFrame para XML utilizando xml.etree.ElementTree e ET.indent
    exatamente conforme a sintaxe ensinada nos slides da Aula 05.
    """
    raiz = ET.Element("contratos_rescindidos", empresa="Imobiliaria_Baggio")

    # Seleciona colunas principais para manter o XML limpo e legível pelo pd.read_xml()
    colunas_xml = [
        "Referência Imóvel", "Referência Contrato", "Status", "Data de Início",
        "Data de Término", "Data de Rescisão", "Tipo de Contrato", "Valor Líquido",
        "Garantia", "Motivo de Rescisão", "Bairro Imóvel", "Meses_Locados", "Quebra_Antecipada"
    ]
    cols_presentes = [c for c in colunas_xml if c in df.columns]

    for _, row in df[cols_presentes].iterrows():
        item_elem = ET.SubElement(raiz, "contrato")
        for col in cols_presentes:
            # Cria tags XML válidas substituindo espaços por underscore
            tag_limpa = (
                col.replace(" ", "_")
                .replace("ê", "e")
                .replace("ó", "o")
                .replace("í", "i")
                .replace("é", "e")
                .replace("ã", "a")
            )
            filho = ET.SubElement(item_elem, tag_limpa)
            val = row[col]
            filho.text = "" if pd.isna(val) else str(val)

    arvore = ET.ElementTree(raiz)
    ET.indent(arvore, space="    ")
    with open(destino_xml, "wb") as arq:
        arvore.write(arq, encoding="utf-8", xml_declaration=True)


def gerar_lotes_teste_lgpd():
    """Lê as planilhas brutas locais, higieniza (LGPD) e gera lotes CSV, JSON e XML."""
    samples_dir = Path("samples_upload")

    # Procura os arquivos brutos na raiz ou na pasta raw_private
    possiveis_imoveis = [
        Path("raw_private/Imóveis-2026-10-25-21-58-44.xlsx - Worksheet.csv"),
        Path("Imóveis-2026-10-25-21-58-44.xlsx - Worksheet.csv")
    ]
    possiveis_contratos = [
        Path("raw_private/Contratos-Rescindidos-2026-10-27-16-03-52.xlsx"),
        Path("Contratos-Rescindidos-2026-10-27-16-03-52.xlsx")
    ]

    caminho_im = next((p for p in possiveis_imoveis if p.exists()), None)
    caminho_ct = next((p for p in possiveis_contratos if p.exists()), None)

    if caminho_im:
        print(f"🔄 Higienizando base de Imóveis ({caminho_im.name})...")
        df_im_raw = pd.read_csv(caminho_im, encoding="utf-8")
        df_im_limpo = higienizar_imoveis(df_im_raw)

        # 1. Salva lote principal (primeiros 2.900 imóveis) em CSV
        df_base_2900 = df_im_limpo.iloc[:2900].copy()
        df_base_2900.to_csv(samples_dir / "lote_imoveis_base_2900.csv", index=False, encoding="utf-8")

        # 2. Cria um lote Delta em JSON (50 imóveis que mudaram de status + 46 novos que entraram)
        df_delta = df_im_limpo.iloc[2850:].copy()
        # Simula mudança de status operacional nos primeiros 10 imóveis sobrepostos
        df_delta.iloc[:10, df_delta.columns.get_loc("Status")] = "Baixado"
        df_delta.to_json(samples_dir / "lote_imoveis_atualizacao_delta.json", orient="records", force_ascii=False, indent=2)

        # 3. Salva base completa limpa (2.946 imóveis)
        df_im_limpo.to_csv(samples_dir / "imoveis_completo_lgpd.csv", index=False, encoding="utf-8")
        print("✅ Lotes de Imóveis gerados em 'samples_upload/' (CSV e JSON).")
    else:
        print("⚠️ Planilha bruta de Imóveis não encontrada na raiz ou em 'raw_private/'.")

    if caminho_ct:
        print(f"🔄 Higienizando base de Contratos ({caminho_ct.name})...")
        xls = pd.ExcelFile(caminho_ct)
        df_ct_raw = pd.read_excel(caminho_ct, sheet_name=xls.sheet_names[0])
        df_ct_limpo = higienizar_contratos(df_ct_raw)

        # Exporta nos 3 formatos exigidos pelo professor (.csv, .json e .xml)
        df_ct_limpo.to_csv(samples_dir / "lote_contratos_maio.csv", index=False, encoding="utf-8")
        df_ct_limpo.to_json(samples_dir / "lote_contratos_maio.json", orient="records", force_ascii=False, indent=2)
        exportar_contratos_para_xml(df_ct_limpo, samples_dir / "lote_contratos_maio.xml")
        print("✅ Lotes de Contratos gerados em 'samples_upload/' nos formatos .CSV, .JSON e .XML!")
    else:
        print("⚠️ Planilha bruta de Contratos não encontrada na raiz ou em 'raw_private/'.")


if __name__ == "__main__":
    construir_estrutura_pastas()
    gerar_lotes_teste_lgpd()
    print("\n🚀 Ambiente pronto! Execute agora: streamlit run app.py")