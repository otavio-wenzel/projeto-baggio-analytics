"""
Mapeamento integral dos esquemas (115 colunas de Imóveis e 77 de Contratos)
e tradutor de rótulos amigáveis para eliminar underscores (_) no front-end.
"""

COLUNAS_SCHEMA_IMOVEIS = (
    "ID", "Filial", "Referência", "Carteira", "Tipo de Angariação", "Tipo", "Status",
    "Tipo de Transação", "Tipo de Venda", "Emissão de NFe", "Dimob", "Condomínio",
    "Condomínio Adm. pela empresa", "Água Adm. pela empresa", "Luz Adm. pela empresa",
    "Tipo de Construção", "Zona", "Zoneamento", "Estado de ocupação", "Tipo de Garagem",
    "Condição do Imóvel", "Topografia", "Apartamentos por Andar", "Elevadores",
    "Ano de Construção", "Finalizado", "Lançado", "Ativo", "Energia", "Água",
    "Administradora", "Valor Bruto", "Valor Líquido", "Delta Valor desde publicado",
    "Desconto de Pontualidade (%)", "Garagens", "Vagas de Garagem", "Total de Quartos",
    "Suítes", "Demisuítes", "Banheiros (inc. suítes)", "Exclusivo", "Publicações",
    "Título", "Descrição", "Informações de Venda", "Foto Principal", "Área Total (m2)",
    "Área Privativa (m2)", "Área Útil (m2)", "Área do Terreno (m2)", "Área Construída (m2)",
    "Área Averbada (m2)", "Área do Escritório (m2)", "Área do Mezanino (m2)",
    "Área do Terraço (m2)", "Área Locável (m2)", "Medidas do Terreno",
    "Taxa Administrativa (%)", "Tx. Adm. Incide Sobre", "Tipo de Intermediação",
    "Valor Intermediação", "Parcelas Intermediação", "Intermediação sobre",
    "Localização", "Uso", "Quitado", "Financiado", "Esquadrias", "Revestimento Externo",
    "Endereço", "Número", "Complemento", "Bairro", "Cidade", "Estado", "CEP",
    "Latitude", "Longitude", "Referência - Entre as ruas", "Referência - Proximo a",
    "Alerta Comercial", "Alerta Gestão", "Angariador", "Data de Angariação",
    "Valor Avaliado", "Valor Esperado pelo Prop.", "Desconto no aluguel (%)",
    "Preço de Venda", "Taxa de Condominio", "IPTU", "FCI", "Valor Vistoria",
    "Seguro Incêndio", "Repasse Garantido", "Meses Garantidos", "Propr. - Tipo",
    "Propr. - Nome | Razão Social", "Propr. - Nome Fantasia", "Propr. - CPF | CPNJ",
    "Propr. - RG | IE", "Propr. - Email Principal", "Propr. - Outros emails",
    "Propr. - Telefone Principal", "Propr. - Contatos adicionais",
    "Propr. - Data de Nascimento", "Propr. - Profissão", "Propr. - Nacionalidade",
    "Propr. - Estado Civil", "Data de Publicação", "Dias desde a publicação",
    "Data de Cadastro", "Data de Baixa", "Motivo de Baixa", "Data de Auditoria"
)

COLUNAS_SCHEMA_CONTRATOS = (
    "ID", "Filial", "Referência Imóvel", "Referência Contrato", "Carteira", "Status",
    "Data de Entrega das Chaves", "Status Financeiro", "Data de Início", "Data de Término",
    "Tipo de Contrato", "Ramo de Atividade", "Valor Bruto", "Valor Líquido",
    "Desconto de Pontualidade (%)", "Dia de Vencimento", "Antecipado",
    "Índice de Reajuste", "Método de Cobrança", "Garantia", "Garantia - Valor",
    "# Parc. Garantia", "R$ Parc. Garantia", "Início Garantia", "Fim Garantia",
    "Resp. Garantia", "# Parc. S.I.", "R$ Parc. S.I.", "Início S.I.", "Fim S.I.",
    "# Parc. S.F.", "R$ Parc. S.F.", "Início S.F.", "Fim S.F.", "Seguradora",
    "% FCI", "Data de Último Reajuste", "Data do Aviso Prévio", "Data de Rescisão",
    "Motivo de Rescisão", "Observações da Rescisão", "Inq - Nome", "Inq - CPF/CNPJ",
    "Bairro Imóvel", "Cidade Imóvel", "Estado Imóvel", "CEP Imóvel",
    "Dias até a contratação", "Observações", "Proprietário - Nome",
    "Proprietário - Documento", "Proprietário - Tipo", "Proprietário - Estado Civil",
    "Proprietário - Nacionalidade", "Proprietário - Profissão", "Proprietário - Email",
    "Proprietário - Telefone Principal", "Proprietário - Data de Nascimento",
    "Proprietário - Carteira", "Imóvel - Rua", "Imóvel - Distrito", "Imóvel - Cidade",
    "Imóvel - Estado", "Proprietário - Rua", "Proprietário - Distrito",
    "Proprietário - Cidade", "Proprietário - Estado", "Inquilino - Nome",
    "Inquilino - Estado Civil", "Inquilino - Nacionalidade", "Inquilino - Profissão",
    "Inquilino - Email", "Inquilino - Telefone Principal", "Inquilino - Rua",
    "Inquilino - Distrito", "Inquilino - Cidade", "Inquilino - Estado"
)

OPCOES_TIPO_IMOVEL = (
    "Apartamento", "Casa Residencial", "Sobrado", "Casa em Condomínio",
    "Loja", "Salas/Conjuntos", "Barracão/Galpão", "Galpão Industrial",
    "Terreno Residencial", "Terreno Comercial"
)

OPCOES_STATUS_IMOVEL = (
    "Alugado", "Divulgado", "Baixado", "Em Desocupação",
    "Em Reparo - Proprietário", "Em Reparo/Manutenção", "Reservado",
    "Carteira Própria", "Preparação de Contrato", "Preparo para Captação", "Interno"
)

OPCOES_REPASSE = (
    "Garantir por alguns meses", "Garantir por todo o contrato", "Não garante repasse"
)

OPCOES_TX_INCIDE = (
    "Somente o Aluguel", "Aluguel e serviços", "Aluguel e serviços - exceto IPTU",
    "Aluguel e Serviços - exceto IPTU e Condomínio", "Aluguel e Serviços - exceto Condomínio"
)

OPCOES_MOTIVO_BAIXA = (
    "Não aplicável (Imóvel Ativo)", "Uso Próprio | Desistência | Reforma",
    "Adm. Particular | Locou Direto", "Vendeu o imóvel", "Auditoria de Carteira",
    "Particular", "Baixa de Proprietário", "Demora na Locação",
    "Imóvel de vitrine | Retirado de divulgação", "Insatisfeito com a Imobiliária"
)

OPCOES_GARANTIA = (
    "Seguro Fiança", "Fiador", "Título de Capitalização", "Carta Fiança", "Sem Garantia"
)

OPCOES_TIPO_CONTRATO = ("Residencial", "Não residencial")

OPCOES_MOTIVO_RESCISAO = (
    "Mudança de Cidade | Desemprego", "Valor | Redução de Custo",
    "Vendeu o Imóvel", "Prop. Solicitou | Reforma | Venda",
    "Encerramento de Atividade Comercial", "Localização",
    "Compra do Imóvel", "Problema de Manutenção", "Particular"
)

BAIRROS_PRINCIPAIS = (
    "Cidade Industrial", "Campo Comprido", "Centro", "Mossunguê", "São Braz",
    "Bigorrilho", "Orleans", "Vila Izabel", "Ecoville", "Portão",
    "Água Verde", "Campina do Siqueira", "Santa Quitéria", "Seminário", "São Francisco"
)

# Dicionário de tradução de variáveis internas (snake_case) para rótulos limpos de negócio
ROTULOS_AMIGAVEIS = {
    "Faixa_Carteira_Prop": "Porte da Carteira do Proprietário",
    "Qtd_Imoveis_Proprietario": "Imóveis por Proprietário",
    "Hash_Proprietario": "Identificador Anônimo (Hash SHA-256)",
    "Meses_Locados": "Tempo de Permanência (Meses)",
    "Quebra_Antecipada": "Quebra Antecipada do Inquilino",
    "Proprietario_Evadiu": "Evasão do Proprietário (Imóvel Baixado)",
    "Valor Líquido_contrato": "Aluguel Líquido do Contrato (R$)",
    "Valor Líquido_imovel": "Aluguel Líquido Cadastrado (R$)",
    "Valor Líquido": "Valor Líquido do Aluguel (R$)",
    "Taxa de Condominio": "Taxa de Condomínio (R$)",
    "Área Total (m2)": "Área Total (m²)",
    "Tx. Adm. Incide Sobre": "Base da Taxa Administrativa",
    "Status_imovel": "Status do Imóvel",
    "Status_contrato": "Status do Contrato",
    "Cluster_KMeans": "Grupo K-Means (Persona)",
    "Cluster_DBSCAN": "Classificação DBSCAN (Densidade)",
    "Outros_Bairros": "Outros Bairros",
    "Aluguel_Medio": "Aluguel Médio (R$)",
    "Condominio_Medio": "Condomínio Médio (R$)",
    "Meses_Medio": "Permanência Média (Meses)",
    "Imoveis_Prop": "Média de Imóveis / Proprietário",
    "Quebra_Pct": "Quebra Antecipada (%)",
    "Evasao_Pct": "Evasão do Proprietário (%)"
}


def formatar_rotulo(nome: str) -> str:
    """Converte qualquer nome técnico ou coluna com underscore (_) em expressão limpa."""
    texto = str(nome)
    if texto in ROTULOS_AMIGAVEIS:
        return ROTULOS_AMIGAVEIS[texto]
    if texto.startswith("Bairro_Agrupado_"):
        return f"Bairro: {texto.replace('Bairro_Agrupado_', '').replace('_', ' ')}"
    if texto.startswith("Garantia_"):
        return f"Garantia: {texto.replace('Garantia_', '').replace('_', ' ')}"
    if texto.startswith("Repasse Garantido_"):
        return f"Repasse: {texto.replace('Repasse Garantido_', '').replace('_', ' ')}"
    if texto.startswith("Tx. Adm. Incide Sobre_"):
        return f"Taxa Adm.: {texto.replace('Tx. Adm. Incide Sobre_', '').replace('_', ' ')}"
    if texto.startswith("Tipo de Contrato_"):
        return f"Contrato: {texto.replace('Tipo de Contrato_', '').replace('_', ' ')}"
    return texto.replace("_", " ")