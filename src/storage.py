import os
import io
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np

from src.schemas import COLUNAS_SCHEMA_IMOVEIS, COLUNAS_SCHEMA_CONTRATOS

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data_store"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
LGPD_HASH_SALT = os.getenv("LGPD_HASH_SALT", "baggio_curitiba_2026_seguro")

PATH_IMOVEIS = DATA_DIR / "imoveis_anonimizados.json"
PATH_CONTRATOS = DATA_DIR / "contratos_anonimizados.json"
CONFIG_FILE = Path("settings.json")

DEFAULT_CONFIG = {
    "theme": "Automático",
    "font_size": 14,
    "auto_save": True
}

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


def carregar_configuracoes() -> dict:
    """Carrega preferências do usuário (tema escuro/claro/automático) via settings.json."""
    if not CONFIG_FILE.exists():
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_CONFIG.copy()


def salvar_configuracoes(novas_opcoes: dict) -> None:
    """Persiste configurações de interface em disco usando escrita segura UTF-8."""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(novas_opcoes, f, indent=4, ensure_ascii=False)


def gerar_hash_proprietario(documento: str) -> str:
    """Criptografa CPF/CNPJ em Hash SHA-256 truncado com Salt de segurança."""
    doc_limpo = str(documento).strip()
    if not doc_limpo or doc_limpo.lower() == "nan":
        return "nao_informado"
    texto_blindado = f"{doc_limpo}_{LGPD_HASH_SALT}"
    return hashlib.sha256(texto_blindado.encode("utf-8")).hexdigest()[:12]


def ler_arquivo_flexivel(uploaded_file) -> pd.DataFrame:
    """Lê arquivos CSV, JSON, XML ou XLSX em memória volátil (RAM)."""
    nome_seguro = Path(uploaded_file.name).name.lower()
    buffer = io.BytesIO(uploaded_file.read())

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
        return pd.read_excel(buffer, sheet_name=xls.sheet_names[0])
    raise ValueError("Formato inválido. Utilize CSV, JSON, XML ou XLSX.")


def identificar_modelo_estrutural(df: pd.DataFrame) -> str:
    """Identifica o modelo da tabela pela interseção de conjuntos de colunas."""
    colunas = set(df.columns)
    if ASSINATURA_IMOVEIS.issubset(colunas):
        return "imoveis"
    if ASSINATURA_CONTRATOS.issubset(colunas):
        return "contratos"
    return "invalido"


def anonimizar_imoveis_lgpd(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza categorias, aplica Hashing SHA-256 no CPF/CNPJ e descarta PII."""
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
    """Recalcula o porte do proprietário sobre a base consolidada via Hash_Proprietario."""
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
    """Padroniza strings e remove todas as colunas sensíveis de contratos."""
    df_clean = df.copy()
    for col in ["Garantia", "Tipo de Contrato", "Motivo de Rescisão", "Status"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().replace("nan", np.nan)
    cols_remover = [c for c in COLUNAS_PII_CONTRATOS if c in df_clean.columns]
    return df_clean.drop(columns=cols_remover)


def salvar_atomico_json(df: pd.DataFrame, caminho_destino: Path) -> None:
    """Grava em arquivo temporário e substitui o oficial (Escrita Atômica)."""
    caminho_temp = caminho_destino.with_suffix(".tmp.json")
    registros = json.loads(df.to_json(orient="records", date_format="iso"))
    with open(caminho_temp, mode="w", encoding="utf-8") as f:
        json.dump(registros, f, ensure_ascii=False, indent=2)
    caminho_temp.replace(caminho_destino)


def carregar_base_persistida(caminho: Path) -> pd.DataFrame:
    """Lê a base JSON persistida em disco."""
    if not caminho.exists():
        return pd.DataFrame()
    with open(caminho, mode="r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


def ingerir_e_persistir_lote(uploaded_file) -> dict:
    """Ingere arquivo em lote, valida estrutura, anonimiza e realiza Upsert sem duplicar."""
    df_bruto = ler_arquivo_flexivel(uploaded_file)
    modelo = identificar_modelo_estrutural(df_bruto)

    if modelo == "invalido":
        raise ValueError("Estrutura de colunas incompatível com os modelos de Imóveis ou Contratos.")

    if modelo == "imoveis":
        df_novo = anonimizar_imoveis_lgpd(df_bruto)
        df_hist = carregar_base_persistida(PATH_IMOVEIS)
        qtd_antes = len(df_hist)
        df_cons = pd.concat([df_hist, df_novo], ignore_index=True)
        df_cons = df_cons.drop_duplicates(subset=["Referência"], keep="last").copy()
        df_cons = recalcular_porte_carteira(df_cons)
        salvar_atomico_json(df_cons, PATH_IMOVEIS)
        novos = max(0, len(df_cons) - qtd_antes)
        return {
            "tipo": "Base de Imóveis",
            "lidos": len(df_bruto),
            "novos": novos,
            "atualizados": len(df_bruto) - novos,
            "total": len(df_cons)
        }

    df_novo = anonimizar_contratos_lgpd(df_bruto)
    df_hist = carregar_base_persistida(PATH_CONTRATOS)
    qtd_antes = len(df_hist)
    df_cons = pd.concat([df_hist, df_novo], ignore_index=True)
    df_cons = df_cons.drop_duplicates(subset=["Referência Contrato"], keep="last").copy()
    salvar_atomico_json(df_cons, PATH_CONTRATOS)
    novos = max(0, len(df_cons) - qtd_antes)
    return {
        "tipo": "Contratos Rescindidos",
        "lidos": len(df_bruto),
        "novos": novos,
        "atualizados": len(df_bruto) - novos,
        "total": len(df_cons)
    }


def inserir_registro_manual(tipo_tabela: str, dados_formulario: dict) -> dict:
    """
    Inclui manualmente um Imóvel ou Contrato preenchendo todas as colunas do esquema,
    aplicando criptografia SHA-256 e descarte LGPD em memória antes da escrita atômica.
    """
    if tipo_tabela == "imovel":
        registro_completo = {col: np.nan for col in COLUNAS_SCHEMA_IMOVEIS}
        registro_completo.update(dados_formulario)
        df_unitario = pd.DataFrame([registro_completo])
        df_curado = anonimizar_imoveis_lgpd(df_unitario)

        df_hist = carregar_base_persistida(PATH_IMOVEIS)
        df_cons = pd.concat([df_hist, df_curado], ignore_index=True)
        df_cons = df_cons.drop_duplicates(subset=["Referência"], keep="last").copy()
        df_cons = recalcular_porte_carteira(df_cons)
        salvar_atomico_json(df_cons, PATH_IMOVEIS)
        return {
            "referencia": dados_formulario.get("Referência"),
            "hash_gerado": df_curado["Hash_Proprietario"].iloc[0],
            "total": len(df_cons)
        }

    registro_completo = {col: np.nan for col in COLUNAS_SCHEMA_CONTRATOS}
    registro_completo.update(dados_formulario)
    df_unitario = pd.DataFrame([registro_completo])
    df_curado = anonimizar_contratos_lgpd(df_unitario)

    df_hist = carregar_base_persistida(PATH_CONTRATOS)
    df_cons = pd.concat([df_hist, df_curado], ignore_index=True)
    df_cons = df_cons.drop_duplicates(subset=["Referência Contrato"], keep="last").copy()
    salvar_atomico_json(df_cons, PATH_CONTRATOS)
    return {
        "referencia": dados_formulario.get("Referência Contrato"),
        "hash_gerado": "Anonimizado (PII Descartada)",
        "total": len(df_cons)
    }


def atualizar_imovel_crud(referencia: str, campos_atualizados: dict) -> bool:
    """
    Atualiza atributos de um imóvel existente (Status, Reentrada na carteira,
    Motivo/Data de Baixa, Aluguel, Condomínio ou Repasse) com escrita atômica.
    """
    df_hist = carregar_base_persistida(PATH_IMOVEIS)
    if df_hist.empty or "Referência" not in df_hist.columns:
        return False

    mascara = df_hist["Referência"].astype(str) == str(referencia)
    if not mascara.any():
        return False

    for col, valor in campos_atualizados.items():
        if col == "Propr. - CPF | CPNJ" and valor:
            df_hist.loc[mascara, "Hash_Proprietario"] = gerar_hash_proprietario(str(valor))
        elif col in df_hist.columns:
            df_hist.loc[mascara, col] = valor

    df_hist = recalcular_porte_carteira(df_hist)
    salvar_atomico_json(df_hist, PATH_IMOVEIS)
    return True


def remover_imovel_crud(referencia: str) -> bool:
    """Remove um imóvel específico pela Referência e recalcula as carteiras."""
    df_hist = carregar_base_persistida(PATH_IMOVEIS)
    if df_hist.empty or "Referência" not in df_hist.columns:
        return False

    qtd_antes = len(df_hist)
    df_filtrado = df_hist[df_hist["Referência"].astype(str) != str(referencia)].copy()
    if len(df_filtrado) == qtd_antes:
        return False

    df_filtrado = recalcular_porte_carteira(df_filtrado)
    salvar_atomico_json(df_filtrado, PATH_IMOVEIS)
    return True