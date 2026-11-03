import io
import os
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env para a memória volátil (RAM)
load_dotenv()

# Diretório local de persistência não-volátil (Disco SSD/HDD)
DATA_DIR = Path(os.getenv("DATA_DIR", "data_store"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
LGPD_HASH_SALT = os.getenv("LGPD_HASH_SALT", "padrao_baggio_lgpd")

def gerar_hash_proprietario(documento: str) -> str:
    texto_blindado = f"{documento.strip()}_{LGPD_HASH_SALT}"
    return hashlib.sha256(texto_blindado.encode("utf-8")).hexdigest()[:12]

PATH_IMOVEIS = DATA_DIR / "imoveis_anonimizados.json"
PATH_CONTRATOS = DATA_DIR / "contratos_anonimizados.json"

# Assinaturas estruturais baseadas em Conjuntos (Sets - Filtro Exclusivo)
ASSINATURA_IMOVEIS = {
    "Referência", "Bairro", "Tipo", "Status", "Valor Líquido",
    "Taxa de Condominio", "Área Total (m2)", "Total de Quartos", "Repasse Garantido"
}

ASSINATURA_CONTRATOS = {
    "Referência Contrato", "Referência Imóvel", "Data de Início",
    "Data de Rescisão", "Data de Término", "Garantia", "Tipo de Contrato"
}

# Listas de colunas sensíveis (PII) para descarte imediato em RAM (LGPD)
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


def ler_arquivo_flexivel(uploaded_file) -> pd.DataFrame:
    """
    Lê arquivos .csv, .json, .xml ou .xlsx em memória volátil (RAM)
    ignorando o nome do arquivo para fins de negócio e prevenindo Path Traversal.
    """
    nome_seguro = Path(uploaded_file.name).name.lower()
    conteudo_bytes = uploaded_file.read()
    buffer = io.BytesIO(conteudo_bytes)

    if nome_seguro.endswith(".csv"):
        try:
            return pd.read_csv(buffer, encoding="utf-8")
        except UnicodeDecodeError:
            buffer.seek(0)
            return pd.read_csv(buffer, sep=";", encoding="latin1")
    elif nome_seguro.endswith(".json"):
        return pd.read_json(buffer)
    elif nome_seguro.endswith(".xml"):
        return pd.read_xml(buffer)
    elif nome_seguro.endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(buffer)
        primeira_aba = xls.sheet_names[0]
        return pd.read_excel(buffer, sheet_name=primeira_aba)
    else:
        raise ValueError("Formato não suportado. Envie arquivos CSV, JSON, XML ou XLSX.")


def identificar_modelo_estrutural(df: pd.DataFrame) -> str:
    """
    Validação de Payload: reconhece se a tabela é de Imóveis ou Contratos
    exclusivamente pela estrutura de colunas usando Conjuntos (Sets).
    """
    colunas_presentes = set(df.columns)
    if ASSINATURA_IMOVEIS.issubset(colunas_presentes):
        return "imoveis"
    if ASSINATURA_CONTRATOS.issubset(colunas_presentes):
        return "contratos"
    return "invalido"


def anonimizar_imoveis_lgpd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica padronização textual, gera o Hash SHA-256 do proprietário
    em memória RAM e descarta todas as colunas PII (LGPD).
    """
    df_clean = df.copy()
    cols_texto = ["Bairro", "Tipo", "Status", "Repasse Garantido", "Tx. Adm. Incide Sobre", "Motivo de Baixa"]
    for col in cols_texto:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().replace("nan", np.nan)

    # Reciclagem do documento em Hash SHA-256 antes do descarte
    if "Propr. - CPF | CPNJ" in df_clean.columns:
        df_clean["Hash_Proprietario"] = (
            df_clean["Propr. - CPF | CPNJ"]
            .astype(str)
            .str.strip()
            .apply(lambda doc: hashlib.sha256(doc.encode("utf-8")).hexdigest()[:12])
        )
    elif "Hash_Proprietario" not in df_clean.columns:
        df_clean["Hash_Proprietario"] = "nao_informado"

    colunas_para_remover = [c for c in COLUNAS_PII_IMOVEIS if c in df_clean.columns]
    df_clean = df_clean.drop(columns=colunas_para_remover)
    return df_clean


def recalcular_porte_carteira(df_imoveis: pd.DataFrame) -> pd.DataFrame:
    """
    Recalcula o porte do investidor sobre a base consolidada usando o Hash_Proprietario,
    garantindo atualização precisa quando novos lotes de imóveis forem enviados.
    """
    df_out = df_imoveis.copy()
    if "Hash_Proprietario" in df_out.columns:
        contagem = df_out["Hash_Proprietario"].value_counts()
        df_out["Qtd_Imoveis_Proprietario"] = df_out["Hash_Proprietario"].map(contagem).fillna(1)
        df_out["Faixa_Carteira_Prop"] = pd.cut(
            df_out["Qtd_Imoveis_Proprietario"],
            bins=[0, 1, 3, 10, 500],
            labels=["1 Imóvel (Pequeno)", "2 a 3 Imóveis", "4 a 10 Imóveis (Médio)", "+10 Imóveis (Grande)"]
        ).astype(str)
    return df_out


def anonimizar_contratos_lgpd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza strings e descarta todas as colunas sensíveis de inquilinos e proprietários.
    """
    df_clean = df.copy()
    cols_texto = ["Garantia", "Tipo de Contrato", "Motivo de Rescisão", "Status"]
    for col in cols_texto:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().replace("nan", np.nan)

    colunas_para_remover = [c for c in COLUNAS_PII_CONTRATOS if c in df_clean.columns]
    df_clean = df_clean.drop(columns=colunas_para_remover)
    return df_clean


def salvar_atomico_json(df: pd.DataFrame, caminho_destino: Path) -> None:
    """
    Escrita Atômica recomendada em aula: grava primeiro em um arquivo temporário (.tmp)
    usando Context Manager (with open e UTF-8) e depois substitui o arquivo oficial.
    """
    caminho_temp = caminho_destino.with_suffix(".tmp.json")
    registros = json.loads(df.to_json(orient="records", date_format="iso"))
    with open(caminho_temp, mode="w", encoding="utf-8") as f:
        json.dump(registros, f, ensure_ascii=False, indent=2)
    caminho_temp.replace(caminho_destino)


def carregar_base_persistida(caminho: Path) -> pd.DataFrame:
    """
    Carrega a base JSON persistida em disco utilizando Context Manager seguro.
    """
    if not caminho.exists():
        return pd.DataFrame()
    with open(caminho, mode="r", encoding="utf-8") as f:
        dados = json.load(f)
    return pd.DataFrame(dados)


def ingerir_e_persistir_lote(uploaded_file) -> dict:
    """
    Pipeline completo de Ingestão:
    1. Lê qualquer formato válido em RAM;
    2. Reconhece o modelo pela estrutura de colunas;
    3. Anonimiza PII (LGPD);
    4. Mescla com o histórico e elimina duplicidades mantendo o registro mais recente (Upsert);
    5. Persiste com escrita atômica.
    """
    df_bruto = ler_arquivo_flexivel(uploaded_file)
    modelo = identificar_modelo_estrutural(df_bruto)

    if modelo == "invalido":
        raise ValueError(
            "A estrutura das colunas não corresponde ao modelo de Imóveis nem ao de Contratos Rescindidos."
        )

    if modelo == "imoveis":
        df_novo = anonimizar_imoveis_lgpd(df_bruto)
        df_historico = carregar_base_persistida(PATH_IMOVEIS)
        qtd_antes = len(df_historico)

        df_consolidado = pd.concat([df_historico, df_novo], ignore_index=True)
        df_consolidado = df_consolidado.drop_duplicates(subset=["Referência"], keep="last").copy()
        df_consolidado = recalcular_porte_carteira(df_consolidado)

        salvar_atomico_json(df_consolidado, PATH_IMOVEIS)
        novos_inseridos = len(df_consolidado) - qtd_antes
        return {
            "tipo": "Base de Imóveis",
            "lidos_no_arquivo": len(df_bruto),
            "novos_absorvidos": max(0, novos_inseridos),
            "duplicados_atualizados": len(df_bruto) - max(0, novos_inseridos),
            "total_persistido": len(df_consolidado)
        }

    else:
        df_novo = anonimizar_contratos_lgpd(df_bruto)
        df_historico = carregar_base_persistida(PATH_CONTRATOS)
        qtd_antes = len(df_historico)

        df_consolidado = pd.concat([df_historico, df_novo], ignore_index=True)
        df_consolidado = df_consolidado.drop_duplicates(subset=["Referência Contrato"], keep="last").copy()

        salvar_atomico_json(df_consolidado, PATH_CONTRATOS)
        novos_inseridos = len(df_consolidado) - qtd_antes
        return {
            "tipo": "Contratos Rescindidos",
            "lidos_no_arquivo": len(df_bruto),
            "novos_absorvidos": max(0, novos_inseridos),
            "duplicados_atualizados": len(df_bruto) - max(0, novos_inseridos),
            "total_persistido": len(df_consolidado)
        }