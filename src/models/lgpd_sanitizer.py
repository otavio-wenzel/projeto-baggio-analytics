import os
import hashlib
from dotenv import load_dotenv
import pandas as pd
import numpy as np

load_dotenv()

LGPD_HASH_SALT = os.getenv("LGPD_HASH_SALT", "baggio_curitiba_2026_seguro")

ASSINATURA_IMOVEIS = {
    "Referência", "Bairro", "Tipo", "Status", "Valor Líquido",
    "Taxa de Condominio", "Área Total (m2)", "Total de Quartos", "Repasse Garantido"
}

ASSINATURA_CONTRATOS = {
    "Referência Contrato", "Referência Imóvel", "Data de Início",
    "Data de Rescisão", "Data de Término", "Garantia", "Tipo de Contrato"
}

COLUNAS_PII_IMOVEIS = [
    "Propr. - Nome | Razão Social", "Propr. - Nome Fantasia", "Propr. - CPF | CPNJ",
    "Propr. - RG | IE", "Propr. - Email Principal", "Propr. - Outros emails",
    "Propr. - Telefone Principal", "Propr. - Contatos adicionais", "Endereço",
    "Número", "Complemento", "CEP", "Latitude", "Longitude",
    "Referência - Entre as ruas", "Referência - Proximo a",
    "Propr. - Data de Nascimento", "Propr. - Profissão",
    "Propr. - Nacionalidade", "Propr. - Estado Civil"
]

COLUNAS_PII_CONTRATOS = [
    "Inq - Nome", "Inq - CPF/CNPJ", "Proprietário - Nome", "Proprietário - Documento",
    "Proprietário - Email", "Proprietário - Telefone Principal", "Inquilino - Nome",
    "Inquilino - Email", "Inquilino - Telefone Principal", "Imóvel - Rua",
    "Proprietário - Rua", "Inquilino - Rua", "CEP Imóvel",
    "Proprietário - Data de Nascimento", "Proprietário - Estado Civil",
    "Proprietário - Nacionalidade", "Proprietário - Profissão",
    "Inquilino - Estado Civil", "Inquilino - Nacionalidade",
    "Inquilino - Profissão", "Observações", "Observações da Rescisão"
]


def gerar_hash_proprietario(documento: str) -> str:
    doc_limpo = str(documento).strip()
    if not doc_limpo or doc_limpo.lower() == "nan":
        return "nao_informado"
    texto_blindado = f"{doc_limpo}_{LGPD_HASH_SALT}"
    return hashlib.sha256(texto_blindado.encode("utf-8")).hexdigest()[:12]


def identificar_modelo_estrutural(df: pd.DataFrame) -> str:
    colunas = set(df.columns)
    if ASSINATURA_IMOVEIS.issubset(colunas):
        return "imoveis"
    if ASSINATURA_CONTRATOS.issubset(colunas):
        return "contratos"
    return "invalido"


def anonimizar_imoveis_lgpd(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    for col in ["Bairro", "Tipo", "Status", "Repasse Garantido", "Tx. Adm. Incide Sobre", "Motivo de Baixa"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().replace("nan", np.nan)

    if "Propr. - CPF | CPNJ" in df_clean.columns:
        df_clean["Hash_Proprietario"] = df_clean["Propr. - CPF | CPNJ"].apply(gerar_hash_proprietario)
    elif "Hash_Proprietario" not in df_clean.columns:
        df_clean["Hash_Proprietario"] = "nao_informado"

    cols_remover = [c for c in COLUNAS_PII_IMOVEIS if c in df_clean.columns]
    return df_clean.drop(columns=cols_remover)


def recalcular_porte_carteira(df_imoveis: pd.DataFrame) -> pd.DataFrame:
    df_out = df_imoveis.copy()
    if "Hash_Proprietario" in df_out.columns and not df_out.empty:
        contagem = df_out["Hash_Proprietario"].value_counts()
        df_out["Qtd_Imoveis_Proprietario"] = df_out["Hash_Proprietario"].map(contagem).fillna(1)
        df_out["Faixa_Carteira_Prop"] = pd.cut(
            df_out["Qtd_Imoveis_Proprietario"],
            bins=[0, 1, 3, 10, 500],
            labels=["1 Imóvel (Pequeno)", "2 a 3 Imóveis", "4 a 10 Imóveis (Médio)", "+10 Imóveis (Grande)"]
        ).astype(str)
    return df_out


def anonimizar_contratos_lgpd(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    for col in ["Garantia", "Tipo de Contrato", "Motivo de Rescisão", "Status"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().replace("nan", np.nan)
    cols_remover = [c for c in COLUNAS_PII_CONTRATOS if c in df_clean.columns]
    return df_clean.drop(columns=cols_remover)