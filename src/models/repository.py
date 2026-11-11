import os
import io
import json
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np

from src.models.schemas import COLUNAS_SCHEMA_IMOVEIS, COLUNAS_SCHEMA_CONTRATOS
from src.models.lgpd_sanitizer import (
    gerar_hash_proprietario,
    identificar_modelo_estrutural,
    anonimizar_imoveis_lgpd,
    anonimizar_contratos_lgpd,
    recalcular_porte_carteira
)

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data_store"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

PATH_IMOVEIS = DATA_DIR / "imoveis_anonimizados.json"
PATH_CONTRATOS = DATA_DIR / "contratos_anonimizados.json"
CONFIG_FILE = Path("settings.json")

DEFAULT_CONFIG = {
    "theme": "Automático",
    "font_size": 14,
    "auto_save": True
}


def carregar_configuracoes() -> dict:
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
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(novas_opcoes, f, indent=4, ensure_ascii=False)


def ler_arquivo_flexivel(uploaded_file) -> pd.DataFrame:
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


def salvar_atomico_json(df: pd.DataFrame, caminho_destino: Path) -> None:
    caminho_temp = caminho_destino.with_suffix(".tmp.json")
    registros = json.loads(df.to_json(orient="records", date_format="iso"))
    with open(caminho_temp, mode="w", encoding="utf-8") as f:
        json.dump(registros, f, ensure_ascii=False, indent=2)
    caminho_temp.replace(caminho_destino)


def carregar_base_persistida(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        return pd.DataFrame()
    with open(caminho, mode="r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


def ingerir_e_persistir_lote(uploaded_file) -> dict:
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