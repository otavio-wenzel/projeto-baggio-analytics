import hashlib
import io
import numpy as np
import pandas as pd


COLUNAS_PII_IMOVEIS = [
    "Propr. - Nome | Razão Social", "Propr. - Nome Fantasia", "Propr. - CPF | CPNJ",
    "Propr. - RG | IE", "Propr. - Email Principal", "Propr. - Outros emails",
    "Propr. - Telefone Principal", "Propr. - Contatos adicionais", "Endereço", "Número", "Complemento"
]

COLUNAS_PII_CONTRATOS = [
    "Inq - Nome", "Inq - CPF/CNPJ", "Proprietário - Nome", "Proprietário - Documento",
    "Proprietário - Email", "Proprietário - Telefone Principal", "Inquilino - Nome",
    "Inquilino - Email", "Inquilino - Telefone Principal", "Imóvel - Rua", "Proprietário - Rua", "Inquilino - Rua"
]

COLUNAS_NUMERICAS_IMOVEIS = [
    "Valor Bruto", "Valor Líquido", "Taxa de Condominio", "Área Total (m2)",
    "Área Útil (m2)", "Total de Quartos", "Vagas de Garagem",
    "Taxa Administrativa (%)", "Meses Garantidos", "Latitude", "Longitude"
]

COLUNAS_NUMERICAS_CONTRATOS = [
    "Valor Bruto", "Valor Líquido", "Dias até a contratação", "Desconto de Pontualidade (%)"
]


def ler_arquivo_upload(uploaded_file) -> pd.DataFrame:
    """Lê arquivos nos formatos exigidos (.csv, .json, .xml) e também .xlsx."""
    nome = uploaded_file.name.lower()
    conteudo = uploaded_file.read()
    buffer = io.BytesIO(conteudo)

    if nome.endswith(".csv"):
        try:
            return pd.read_csv(buffer, encoding="utf-8")
        except UnicodeDecodeError:
            buffer.seek(0)
            return pd.read_csv(buffer, encoding="latin-1", sep=None, engine="python")
    elif nome.endswith(".json"):
        return pd.read_json(buffer)
    elif nome.endswith(".xml"):
        return pd.read_xml(buffer)
    elif nome.endswith(".xlsx") or nome.endswith(".xls"):
        xls = pd.ExcelFile(buffer)
        return pd.read_excel(buffer, sheet_name=xls.sheet_names[0])
    else:
        raise ValueError("Formato não suportado. Envie arquivos .csv, .json, .xml ou .xlsx.")


def converter_numeros_br(df: pd.DataFrame, colunas: list) -> pd.DataFrame:
    """Converte strings numéricas no formato brasileiro ('3.500,00') para float."""
    df = df.copy()
    for col in colunas:
        if col in df.columns and df[col].dtype == "object":
            serie_limpa = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(serie_limpa, errors="coerce")
        elif col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def higienizar_imoveis(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza textos, converte números e aplica anonimização LGPD com reciclagem de carteira."""
    df = converter_numeros_br(df, COLUNAS_NUMERICAS_IMOVEIS)

    for col_txt in ["Bairro", "Tipo", "Status", "Repasse Garantido", "Tx. Adm. Incide Sobre", "Motivo de Baixa"]:
        if col_txt in df.columns:
            df[col_txt] = df[col_txt].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan})

    # Reciclagem de atributo sensível antes do descarte LGPD
    if "Propr. - CPF | CPNJ" in df.columns:
        contagem = df["Propr. - CPF | CPNJ"].value_counts()
        df["Qtd_Imoveis_Proprietario"] = df["Propr. - CPF | CPNJ"].map(contagem).fillna(1)
        df["Hash_Proprietario"] = (
            df["Propr. - CPF | CPNJ"]
            .astype(str)
            .apply(lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()[:12])
        )
    elif "Hash_Proprietario" in df.columns:
        contagem = df["Hash_Proprietario"].value_counts()
        df["Qtd_Imoveis_Proprietario"] = df["Hash_Proprietario"].map(contagem).fillna(1)
    else:
        df["Qtd_Imoveis_Proprietario"] = 1

    df["Faixa_Carteira_Prop"] = pd.cut(
        df["Qtd_Imoveis_Proprietario"],
        bins=[0, 1, 3, 10, 5000],
        labels=["1 Imóvel (Pequeno)", "2 a 3 Imóveis", "4 a 10 Imóveis (Médio)", "+10 Imóveis (Grande)"]
    ).astype(str)

    colunas_remover = [c for c in COLUNAS_PII_IMOVEIS if c in df.columns]
    return df.drop(columns=colunas_remover)


def higienizar_contratos(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza contratos rescindidos, calcula variáveis temporais e aplica descarte LGPD."""
    df = converter_numeros_br(df, COLUNAS_NUMERICAS_CONTRATOS)

    for col_txt in ["Garantia", "Tipo de Contrato", "Motivo de Rescisão", "Bairro Imóvel"]:
        if col_txt in df.columns:
            df[col_txt] = df[col_txt].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan})

    for col_dt in ["Data de Início", "Data de Rescisão", "Data de Término"]:
        if col_dt in df.columns:
            df[col_dt] = pd.to_datetime(df[col_dt], errors="coerce")

    if "Data de Rescisão" in df.columns and "Data de Início" in df.columns:
        df["Meses_Locados"] = ((df["Data de Rescisão"] - df["Data de Início"]).dt.days / 30.0).round(2)

    if "Data de Rescisão" in df.columns and "Data de Término" in df.columns:
        df["Quebra_Antecipada"] = np.where(df["Data de Rescisão"] < df["Data de Término"], 1, 0)

    # Converte datas de volta para string ISO para persistência limpa em SQLite/JSON/CSV
    for col_dt in ["Data de Início", "Data de Rescisão", "Data de Término"]:
        if col_dt in df.columns:
            df[col_dt] = df[col_dt].dt.strftime("%Y-%m-%d")

    colunas_remover = [c for c in COLUNAS_PII_CONTRATOS if c in df.columns]
    return df.drop(columns=colunas_remover)


def reconciliar_upsert(df_atual: pd.DataFrame, df_novo: pd.DataFrame, chave_primaria: str) -> tuple:
    """
    Executa a reconciliação (Upsert) usando Conjuntos (Sets - Aula 02).
    Retorna (df_consolidado, relatorio_delta_dict).
    """
    if chave_primaria not in df_novo.columns:
        raise ValueError(f"A coluna chave '{chave_primaria}' não foi encontrada no arquivo enviado.")

    df_novo = df_novo.drop_duplicates(subset=[chave_primaria], keep="last").copy()
    df_novo[chave_primaria] = df_novo[chave_primaria].astype(str).str.strip()

    if df_atual.empty or chave_primaria not in df_atual.columns:
        relatorio = {
            "total_anterior": 0,
            "lote_enviado": len(df_novo),
            "novos_inseridos": len(df_novo),
            "atualizados": 0,
            "mudancas_status": 0,
            "total_consolidado": len(df_novo)
        }
        return df_novo, relatorio

    df_atual = df_atual.copy()
    df_atual[chave_primaria] = df_atual[chave_primaria].astype(str).str.strip()

    # Operações de Conjuntos (Aula 02): Diferença (-), Interseção (&) e União (|)
    set_atual = set(df_atual[chave_primaria].dropna())
    set_novo = set(df_novo[chave_primaria].dropna())

    chaves_novas = set_novo - set_atual
    chaves_comuns = set_novo & set_atual

    # Audita quantos imóveis/contratos existentes mudaram de Status (ex: Alugado -> Baixado)
    mudancas_status = 0
    if "Status" in df_atual.columns and "Status" in df_novo.columns and chaves_comuns:
        mapa_status_antigo = df_atual.set_index(chave_primaria)["Status"].to_dict()
        mapa_status_novo = df_novo.set_index(chave_primaria)["Status"].to_dict()
        for k in chaves_comuns:
            if mapa_status_antigo.get(k) != mapa_status_novo.get(k):
                mudancas_status += 1

    # Concatena e mantém a versão mais recente ('last'), atualizando quem mudou e preservando o histórico
    df_consolidado = pd.concat([df_atual, df_novo], ignore_index=True)
    df_consolidado = df_consolidado.drop_duplicates(subset=[chave_primaria], keep="last").reset_index(drop=True)

    # Recalcula tamanho da carteira do proprietário caso seja a base de imóveis
    if chave_primaria == "Referência" and "Hash_Proprietario" in df_consolidado.columns:
        contagem = df_consolidado["Hash_Proprietario"].value_counts()
        df_consolidado["Qtd_Imoveis_Proprietario"] = df_consolidado["Hash_Proprietario"].map(contagem).fillna(1)

    relatorio = {
        "total_anterior": len(set_atual),
        "lote_enviado": len(set_novo),
        "novos_inseridos": len(chaves_novas),
        "atualizados": len(chaves_comuns),
        "mudancas_status": mudancas_status,
        "total_consolidado": len(df_consolidado)
    }
    return df_consolidado, relatorio