import json
import re
import sqlite3
import datetime
from pathlib import Path
import pandas as pd

# Diretório seguro de armazenamento (Memória Não-Volátil)
DATA_DIR = Path("data_store")
DATA_DIR.mkdir(exist_ok=True)

CONFIG_FILE = Path("settings.json")
LOG_FILE = DATA_DIR / "audit_log.txt"

DEFAULT_CONFIG = {
    "storage_engine": "sqlite",  # Opções: 'sqlite' ou 'filesystem'
    "sqlite_db_name": "imobiliaria_baggio.db",
    "filesystem_format": "csv",  # Opções: 'csv' ou 'json'
    "auto_backup": True,
    "theme": "light"
}


def sanitizar_nome_arquivo(nome: str) -> str:
    """Previne vulnerabilidades de Path Traversal (Aula 05)."""
    nome_limpo = Path(nome).name
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", nome_limpo)


def registrar_log_auditoria(mensagem: str) -> None:
    """Registra eventos operacionais em arquivo TXT usando modo 'a' (Append) e UTF-8."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {mensagem}\n")


def carregar_configuracoes() -> dict:
    """Carrega ou inicializa o arquivo settings.json conforme padrão da Aula 05."""
    if not CONFIG_FILE.exists():
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
        return DEFAULT_CONFIG.copy()
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        dados = json.load(f)
    # Uso seguro de .get() para evitar KeyError (Aula 02)
    config_final = {}
    for chave, valor_padrao in DEFAULT_CONFIG.items():
        config_final[chave] = dados.get(chave, valor_padrao)
    return config_final


def salvar_configuracoes( novas_configs: dict) -> None:
    """Realiza Escrita Atômica no arquivo de configurações JSON."""
    temp_file = CONFIG_FILE.with_suffix(".tmp")
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(novas_configs, f, indent=4, ensure_ascii=False)
    temp_file.replace(CONFIG_FILE)
    registrar_log_auditoria(f"Configuração alterada: engine={novas_configs.get('storage_engine')}")


class StorageRepository:
    """
    Camada de Persistência Desacoplada.
    Qualquer leitura ou escrita de Imóveis e Contratos passa por aqui,
    independentemente de o banco ser SQLite, CSV/JSON local ou SQL externo.
    """

    def __init__(self):
        self.config = carregar_configuracoes()
        self.engine = self.config.get("storage_engine", "sqlite")
        self.db_path = DATA_DIR / sanitizar_nome_arquivo(self.config.get("sqlite_db_name", "imobiliaria_baggio.db"))

    def salvar_tabela(self, nome_tabela: str, df: pd.DataFrame) -> None:
        tabela_segura = sanitizar_nome_arquivo(nome_tabela)
        if self.engine == "sqlite":
            with sqlite3.connect(self.db_path) as conn:
                df.to_sql(tabela_segura, conn, if_exists="replace", index=False)
        else:
            formato = self.config.get("filesystem_format", "csv")
            alvo = DATA_DIR / f"{tabela_segura}.{formato}"
            temp = DATA_DIR / f"{tabela_segura}.{formato}.tmp"
            if formato == "json":
                df.to_json(temp, orient="records", force_ascii=False, indent=2)
            else:
                df.to_csv(temp, index=False, encoding="utf-8")
            # Escrita Atômica: substitui o original apenas após concluir a gravação
            temp.replace(alvo)

        registrar_log_auditoria(
            f"Tabela '{tabela_segura}' persistida via motor '{self.engine}' ({len(df)} registros)."
        )

    def carregar_tabela(self, nome_tabela: str) -> pd.DataFrame:
        tabela_segura = sanitizar_nome_arquivo(nome_tabela)
        if self.engine == "sqlite":
            if not self.db_path.exists():
                return pd.DataFrame()
            try:
                with sqlite3.connect(self.db_path) as conn:
                    return pd.read_sql_query(f"SELECT * FROM [{tabela_segura}]", conn)
            except Exception:
                return pd.DataFrame()
        else:
            formato = self.config.get("filesystem_format", "csv")
            alvo = DATA_DIR / f"{tabela_segura}.{formato}"
            if not alvo.exists():
                # Fallback caso o formato tenha acabado de ser trocado na interface
                alternativo = DATA_DIR / f"{tabela_segura}.{'csv' if formato == 'json' else 'json'}"
                if not alternativo.exists():
                    return pd.DataFrame()
                alvo = alternativo
            if alvo.suffix == ".json":
                return pd.read_json(alvo, orient="records")
            return pd.read_csv(alvo, encoding="utf-8")

    def ler_historico_logs(self, limite_linhas: int = 30) -> list:
        if not LOG_FILE.exists():
            return ["Nenhum evento registrado ainda."]
        linhas = []
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for linha in f:
                linhas.append(linha.strip())
        return linhas[-limite_linhas:][::-1]