# Baggio Intelligence | Analytics, Governança LGPD & Machine Learning

Aplicação web interativa desenvolvida em Python e Streamlit para diagnóstico operacional, retenção de carteira imobiliária, governança de dados sob a Lei Geral de Proteção de Dados (LGPD) e modelagem preditiva com aprendizado de máquina supervisionado e não supervisionado.

* **Disciplina:** Tópicos Especiais em Software
* **Professor:** Rhafael Freitas da Costa - [@rhafael-costa-dev](https://github.com/rhafael-costa-dev)
* **Arquitetura:** Padrão MVC (*Model-View-Controller*) com Princípio da Responsabilidade Única (SRP)

---

## Sumário

1. [Contexto de Negócio e Objetivos do Projeto](#1-contexto-de-negócio-e-objetivos-do-projeto)
2. [Arquitetura de Software (Padrão MVC)](#2-arquitetura-de-software-padrão-mvc)
3. [Estruturas de Dados, Segurança de I/O e Persistência Não-Volátil](#3-estruturas-de-dados-segurança-de-io-e-persistência-não-volátil)
4. [Governança LGPD e Anonimização em Memória Volátil (RAM)](#4-governança-lgpd-e-anonimização-em-memória-volátil-ram)
5. [Engenharia de Dados e Saneamento Estatístico (IQR)](#5-engenharia-de-dados-e-saneamento-estatístico-iqr)
6. [Catálogo de Análises Exploratórias Interativas (EDA)](#6-catálogo-de-análises-exploratórias-interativas-eda)
7. [Laboratório de Machine Learning Supervisionado](#7-laboratório-de-machine-learning-supervisionado)
8. [Pipeline Não Supervisionado (Clusterização, Anomalias e PCA)](#8-pipeline-não-supervisionado-clusterização-anomalias-e-pca)
9. [Requisitos, Configuração de Ambiente e Execução](#9-requisitos-configuração-de-ambiente-e-execução)
10. [Guia de Operação e Manutenção da Base Local](#10-guia-de-operação-e-manutenção-da-base-local)
11. [Matriz de Responsabilidades para Defesa de Código](#11-matriz-de-responsabilidades-para-defesa-de-código)

---

## 1. Contexto de Negócio e Objetivos do Projeto

O sistema **Baggio Intelligence** soluciona o desafio de evasão de proprietários (baixa de imóveis da carteira administrada) e de rescisão antecipada de contratos de locação por inquilinos no mercado imobiliário de Curitiba/PR. A plataforma opera sobre duas bases relacionais complementares:

1. **Base de Imóveis da Carteira (115 colunas no esquema bruto):** Cadastro estrutural, financeiro, geográfico e de políticas operacionais (como Repasse Garantido e Base de Incidência da Taxa Administrativa) dos imóveis geridos pela imobiliária.
2. **Base de Contratos Rescindidos (77 colunas no esquema bruto):** Histórico de locações encerradas, contendo datas de início, término previsto, rescisão efetiva, modalidade de garantia locatícia e motivos de saída.

### Principais Entregas da Aplicação
* **Ingestão Multiformato com Reconhecimento Estrutural:** Upload flexível de arquivos `.csv`, `.json`, `.xml`, `.xlsx` e `.xls` que identifica automaticamente a qual tabela o arquivo pertence analisando a assinatura de suas colunas, sem depender do nome do arquivo.
* **Curadoria LGPD em Tempo Real:** Anonimização irreversível de documentos via criptografia SHA-256 com *Salt* e descarte imediato de dados pessoais sensíveis (PII) ainda na memória RAM antes de qualquer gravação em disco.
* **Atualização Incremental (*Upsert*) e CRUD Ciclo de Vida:** Absorção de lotes mensais com deduplicação automática pela chave de negócio, além de formulários discretos para inclusão avulsa, baixa e reentrada de imóveis na carteira ativa.
* **Visualizações Dinâmicas e Compositor de Sessão:** Nove módulos analíticos independentes com filtros customizáveis, detector matemático de picos sazonais, construtor livre de cruzamentos e modo de ampliação responsiva.
* **Machine Learning Configurável e Retreinamento Dinâmico:** Modelos supervisionados de Regressão (validação cruzada K-Fold contra *Baseline*) e Classificação de Risco, Simulador Executivo de novos contratos e segmentação não supervisionada (K-Means, DBSCAN, PCA e Ward) que se reajustam automaticamente sempre que novos dados são inseridos.

---

## 2. Arquitetura de Software (Padrão MVC)

O código-fonte foi projetado em **14 módulos Python independentes**, distribuídos nas camadas **Model**, **Controller** e **View** para garantir baixo acoplamento, alta coesão e ausência de regras de negócio misturadas à interface gráfica:

```text
projeto-baggio/
│
├── data_store/                          # Persistência JSON não-volátil (Ignorada no Git)
├── settings.json                        # Preferências de tema do usuário (Criado automaticamente)
├── .env                                 # Variáveis de ambiente locais (Ignorado no Git)
├── .env.example                         # Modelo público de variáveis de ambiente
├── .gitignore                           # Regras de bloqueio de segurança e proteção LGPD
├── requirements.txt                     # Dependências do ecossistema Python
├── README.md                            # Documentação técnica e guia do projeto
├── app.py                               # Ponto de Entrada e Roteador Principal (55 linhas)
│
└── src/
    ├── __init__.py
    │
    ├── models/                          # CAMADA MODEL (Esquemas, LGPD e Persistência I/O)
    │   ├── __init__.py
    │   ├── schemas.py                   # Constantes imutáveis, catálogos e tradutor de rótulos
    │   ├── lgpd_sanitizer.py            # Validação estrutural (Sets), SHA-256 e descarte PII
    │   └── repository.py                # Leitura RAM, escrita atômica JSON, Upsert e CRUD
    │
    ├── controllers/                     # CAMADA CONTROLLER (Regras Analíticas e Motores de IA)
    │   ├── __init__.py
    │   ├── data_processor.py            # Cruzamento relacional, cerca IQR, picos e tema Matplotlib
    │   ├── eda_temporal_ctrl.py         # Gráficos A1 (Escalada), A2 (Sazonalidade) e A3 (Motivos)
    │   ├── eda_politicas_ctrl.py        # Gráficos A4 a A9 (Repasse, Taxa, Carteira, Bairros, IQR, Livre)
    │   ├── ml_supervisionado_ctrl.py    # Matriz de features, Regressores (K-Fold) e Classificadores
    │   └── ml_clusters_ctrl.py          # Pipeline Não Supervisionado (K-Means, DBSCAN, PCA e Ward)
    │
    └── views/                           # CAMADA VIEW (Componentes Visuais e Telas Streamlit)
        ├── __init__.py
        ├── ui_components.py             # CSS Global (Claro/Escuro), ampliador de gráficos e Sidebar
        ├── dashboard_view.py            # Tela 1: Visão Geral & Indicadores Executivos
        ├── ingestao_view.py             # Tela 2: Upload em Lote, Cadastro Manual e CRUD Discreto
        ├── eda_view.py                  # Tela 3: Explorador de Análises & Compositor de Sessão
        ├── ml_view.py                   # Tela 4: Laboratório ML & Simulador Executivo
        └── clusters_view.py             # Tela 5: Segmentação de Personas & Auditoria de Anomalias
```

### Papel de Cada Camada
* **Ponto de Entrada (`app.py`):** Inicializa as chaves de estado no `st.session_state`, carrega as bases persistidas e utiliza a estrutura `match st.session_state.rota_atual:` para delegar a renderização à View correspondente em apenas 55 linhas.
* **Camada Model (`src/models/`):** Detém as definições de domínio, a validação estrutural das planilhas, a criptografia LGPD e todas as operações de leitura e escrita em disco.
* **Camada Controller (`src/controllers/`):** Processa os DataFrames do Pandas, executa os cálculos estatísticos, gera os objetos `Figure` do Matplotlib/Seaborn e treina os algoritmos do Scikit-Learn/SciPy sem invocar nenhum comando do Streamlit.
* **Camada View (`src/views/`):** Renderiza os seletores, *sliders*, formulários e painéis na tela, repassando os parâmetros escolhidos pelo usuário para os Controladores.

---

## 3. Estruturas de Dados, Segurança de I/O e Persistência Não-Volátil

O projeto aplica diretamente os fundamentos de estruturas de dados e manipulação segura de arquivos em Python:

### Aplicação das Estruturas Nativas do Python
* **Tuplas (`tuple` — Imutáveis e Ordenadas):** Utilizadas em `src/models/schemas.py` (`COLUNAS_SCHEMA_IMOVEIS`, `COLUNAS_SCHEMA_CONTRATOS`, `OPCOES_GARANTIA`, `BAIRROS_PRINCIPAIS`, etc.) para blindar os esquemas de colunas e domínios fixos contra modificações acidentais durante a execução.
* **Conjuntos (`set` — Mutáveis e Sem Duplicatas):** Aplicados em `src/models/lgpd_sanitizer.py` (`ASSINATURA_IMOVEIS` e `ASSINATURA_CONTRATOS`) por meio do método `.issubset(set(df.columns))` para validar matematicamente a estrutura da tabela enviada no upload em complexidade $O(1)$.
* **Listas (`list` — Mutáveis e Ordenadas):** Empregadas nos catálogos de navegação (`CATALOGO_ANALISES_EDA`, `CATALOGO_SUBMODULOS_ML`, `CATALOGO_SUBMODULOS_CLUSTER`), na filtragem dinâmica de colunas de PII e na fila do compositor de sessão.
* **Dicionários (`dict` — Pares Chave-Valor):** Utilizados no tradutor semântico `ROTULOS_AMIGAVEIS` (que elimina *underscores* e nomes técnicos da interface), no arquivo de preferências `settings.json` e nos payloads de retorno das operações de ingestão e métricas de Machine Learning.

### Práticas de Segurança e Persistência em Disco (`src/models/repository.py`)
1. **Sanitização de Nomes contra *Path Traversal*:** Na função `ler_arquivo_flexivel()`, o nome do arquivo enviado pelo usuário é isolado com `Path(uploaded_file.name).name.lower()` para extrair apenas a extensão e ler os bytes em um buffer de memória volátil (`io.BytesIO`), jamais utilizando o nome informado pelo usuário para gravar caminhos no servidor.
2. **Fallback Automático de Codificação:** Arquivos `.csv` são lidos prioritariamente em `utf-8`; caso ocorra `UnicodeDecodeError` (comum em planilhas exportadas no Brasil), o ponteiro do buffer retorna ao início (`buffer.seek(0)`) e realiza a leitura automática com separador `;` e codificação `latin1`.
3. **Validação de Payload em Memória RAM:** Antes de processar qualquer regra de negócio, `identificar_modelo_estrutural()` verifica se o DataFrame contém todas as colunas obrigatórias de Imóveis ou de Contratos, rejeitando com `ValueError` qualquer arquivo incompatível.
4. **Escrita Atômica (Prevenção contra Corrupção de Dados):** Na função `salvar_atomico_json()`, os dados nunca sobrescrevem diretamente o arquivo oficial. O sistema grava primeiro em um arquivo temporário (`.tmp.json`) com `with open(..., encoding="utf-8")` e, somente após o fechamento íntegro do buffer, executa `caminho_temp.replace(caminho_destino)`, garantindo atomicidade mesmo em caso de queda de energia.
5. **Deduplicação Incremental (*Upsert*):** A cada novo envio em lote ou cadastro manual, o sistema concatena a base histórica com o novo lote (`pd.concat`) e executa `drop_duplicates(subset=["Referência"], keep="last")` para imóveis ou `drop_duplicates(subset=["Referência Contrato"], keep="last")` para contratos, assegurando que envios repetidos do mesmo mês atualizem os registros sem gerar duplicatas.
6. **Persistência de Preferências (`settings.json`):** A função `carregar_configuracoes()` verifica a existência do arquivo `settings.json`; caso não exista na máquina local, recria-o automaticamente a partir do dicionário `DEFAULT_CONFIG` e salva as alterações de tema visual via `salvar_configuracoes()`.

---

## 4. Governança LGPD e Anonimização em Memória Volátil (RAM)

Nenhuma informação pessoal identificável (PII) de proprietários ou inquilinos é gravada na pasta `data_store/`. O módulo `src/models/lgpd_sanitizer.py` executa a curadoria completa ainda na memória volátil:

* **Pseudonimização Irreversível com Hash SHA-256 + *Salt*:**
  O documento do proprietário (`Propr. - CPF | CPNJ`) é combinado a uma chave secreta carregada via variável de ambiente (`LGPD_HASH_SALT`) e criptografado pelo algoritmo SHA-256 truncado nos 12 primeiros caracteres hexadecimais:

$$\text{Hash Proprietario} = \text{SHA256}(\text{Documento} + \text{Salt})[0:12]$$

* **Preservação da Inteligência Analítica (Porte do Investidor):**
  Antes e após cada consolidação da base de imóveis, a função `recalcular_porte_carteira()` agrupa os registros pelo `Hash_Proprietario` para calcular o número exato de imóveis pertencentes ao mesmo investidor (`Qtd_Imoveis_Proprietario`) e classificá-lo em quatro faixas (`Faixa_Carteira_Prop`: *1 Imóvel (Pequeno)*, *2 a 3 Imóveis*, *4 a 10 Imóveis (Médio)* e *+10 Imóveis (Grande)*).
* **Descarte Definitivo de Colunas Sensíveis (`drop`):**
  * **Na Base de Imóveis (20 colunas removidas):** Razão Social, Nome Fantasia, CPF/CNPJ original, RG/IE, E-mails, Telefones, Endereço, Número, Complemento, CEP, Latitude, Longitude, Referências de ruas, Data de Nascimento, Profissão, Nacionalidade e Estado Civil.
  * **Na Base de Contratos (22 colunas removidas):** Nomes, Documentos, E-mails, Telefones, Endereços/Ruas, CEPs, Datas de Nascimento, Estado Civil, Nacionalidade e Profissão (tanto de proprietários quanto de inquilinos), além dos campos de texto livre `Observações` e `Observações da Rescisão`.

---

## 5. Engenharia de Dados e Saneamento Estatístico (IQR)

No módulo `src/controllers/data_processor.py`, a função `preparar_base_cruzada()` executa o relacionamento entre as duas bases anonimizadas e prepara as variáveis-alvo para análise e Machine Learning:

1. **Cruzamento Relacional (`Inner Join`):** Realiza o `pd.merge` entre `df_contratos` e `df_imoveis` conectando a chave estrangeira `Referência Imóvel` à chave primária `Referência`.
2. **Engenharia das Variáveis-Alvo (*Targets*):**
   * **Tempo de Permanência Contínuo (`Meses_Locados`):**

$$\text{Meses Locados} = \frac{\text{Data de Rescisão} - \text{Data de Início}}{30,0}$$

   * **Quebra Antecipada do Inquilino (`Quebra_Antecipada` binária $0 \text{ ou } 1$):** Assume valor $1$ quando `Data de Rescisão < Data de Término`, e $0$ caso contrário.
   * **Evasão do Proprietário (`Proprietario_Evadiu` binária $0 \text{ ou } 1$):** Assume valor $1$ quando o status atual do imóvel na carteira é `"Baixado"`, e $0$ caso contrário.
3. **Saneamento de *Outliers* pela Cerca Interquartil (IQR):**
   Filtra registros inconsistentes (`Meses_Locados` $\le 0$) e calcula o primeiro quartil ($Q_1$, percentil 25%) e o terceiro quartil ($Q_3$, percentil 75%), definindo a amplitude interquartil $\text{IQR} = Q_3 - Q_1$ e o limite superior de corte configurável por $k$ (padrão $k = 1,5$):

$$\text{Limite Superior} = Q_3 + k \cdot (Q_3 - Q_1)$$

   Apenas contratos dentro do limite estatístico compõem o DataFrame saneado (`df_limpo`), evitando distorções nas médias e nos modelos preditivos.

---

## 6. Catálogo de Análises Exploratórias Interativas (EDA)

A tela **Explorador de Análises Independentes & Compositor de Sessão** (`src/views/eda_view.py`) disponibiliza 9 módulos analíticos implementados em `eda_temporal_ctrl.py` e `eda_politicas_ctrl.py`, permitindo agrupar múltiplos gráficos na mesma sessão e ampliar qualquer visualização:

| Código | Nome da Análise | Principais Controles Interativos e Método Estatístico |
| :--- | :--- | :--- |
| **A1** | **Escalada Anual de Retiradas** | Filtro de intervalo de anos, tipos de imóvel, alternância entre contagem absoluta ou participação (%) e escolha de gráfico (Barras Verticais, Linha com Marcadores ou Pizza). |
| **A2** | **Sazonalidade Mensal (Picos Automáticos)** | Filtro de anos, recorte de meses ($1$ a $12$), seleção de motivos de baixa e detector automático de janelas de pico sazonal onde o volume mensal supera $\mu + k \cdot \sigma$ (média mais $k$ desvios-padrão configuráveis pelo *slider* de sensibilidade). |
| **A3** | **Ranking de Motivos de Evasão / Rescisão** | Alterna a fonte de dados entre *Imóveis (Motivo de Baixa do Proprietário)* e *Contratos (Motivo de Rescisão do Inquilino)*, com filtro temporal, *Top N* motivos e unidade em percentual (%) ou volume absoluto. |
| **A4** | **Política Interna: Repasse Garantido** | Compara a retenção (*Alugado*) versus evasão (*Baixado*) por regra de Repasse Garantido, filtrando por faixa de valor líquido do aluguel (R$) e tipos de imóvel. |
| **A5** | **Política Interna: Base da Taxa Adm.** | Avalia o impacto da regra de incidência da Taxa Administrativa sobre a taxa de evasão (% Baixado) ou sobre o aluguel líquido médio (R$), com corte mínimo de representatividade amostral. |
| **A6** | **Perfil do Investidor (Porte da Carteira)** | Cruza as faixas de carteira calculadas via Hash SHA-256 com o percentual de retenção/evasão, aluguel líquido médio ou condomínio médio (filtrando até o percentil $P_{95}$). |
| **A7** | **Geografia de Evasão por Bairro** | Diagnóstico nos *Top N* maiores bairros da carteira comparando a taxa de evasão (% Baixado com linha de referência de 35%) ou as medianas de aluguel e condomínio. |
| **A8** | **Garantias Locatícias, IQR & Correlação** | Três modos visuais: (1) Permanência média por modalidade de garantia; (2) Auditoria *Boxplot* lado a lado (Base Bruta vs. Base Saneada pela cerca $k \times \text{IQR}$); (3) Mapa de Calor da Matriz de Correlação Linear de Pearson ($r$). |
| **A9** | **Construtor Livre de Cruzamentos** | Permite ao usuário cruzar livremente qualquer variável categórica (Eixo X) com qualquer métrica numérica (Eixo Y), escolhendo entre Barras Verticais, Barras Horizontais, *Boxplot*, Dispersão (*Scatter*) ou Pizza, com filtro de *outliers* por percentil ($P_{85}$ a $P_{100}$). |

---

## 7. Laboratório de Machine Learning Supervisionado

Implementado em `src/controllers/ml_supervisionado_ctrl.py` e `src/views/ml_view.py`, o laboratório opera sobre a matriz preditora $X$ construída por `preparar_matriz_features()`, que combina variáveis numéricas (imputadas pela mediana) e variáveis categóricas transformadas via *One-Hot Encoding* (`pd.get_dummies(..., drop_first=True)`).

### 7.1. Regressão Supervisionada & Validação Cruzada K-Fold
* **Variável Alvo ($y$):** `Meses_Locados` (tempo contínuo de permanência do contrato em meses).
* **Algoritmos Disponíveis:**
  1. `Baseline (Média do Treino)` via `DummyRegressor(strategy="mean")` — régua mínima de comparação.
  2. `Regressão Linear (OLS)` via pipeline com `StandardScaler()` e `LinearRegression()`.
  3. `Regressão Ridge (Penalidade L2)` via pipeline com `StandardScaler()` e `Ridge(alpha=alpha)`.
  4. `Regressão Lasso (Penalidade L1)` via pipeline com `StandardScaler()` e `Lasso(alpha=alpha)`.
  5. `Árvore de Decisão (Regressor)` via `DecisionTreeRegressor(max_depth=max_depth)`.
  6. `Floresta Aleatória (Random Forest)` via `RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)`.
* **Hiperparâmetros Configuráveis pelo Usuário:** Penalidade de regularização $\alpha$ ($0,1$ a $20,0$), profundidade máxima das árvores ($2$ a $15$), número de árvores na floresta ($20$ a $300$) e proporção da divisão de teste `test_size` (15% a 35%).
* **Métricas e Diagnóstico Automático:** Calcula o RMSE médio e desvio-padrão ($\pm$) na validação cruzada `KFold` (`scoring="neg_root_mean_squared_error"`), MAE no teste, RMSE no teste, Coeficiente de Determinação $R^2$ (%) e emite o diagnóstico automático de ajuste (*Subajuste / Alta Variância*, *Indício de Sobreajuste* ou *Bom Ajuste*), acompanhado do gráfico *Real vs. Predito* e do ranking de impacto das variáveis (`coef_` ou `feature_importances_`).

### 7.2. Classificadores Supervisionados de Risco
* **Variáveis Alvo Configuráveis ($y$ binário $0 \text{ ou } 1$):** `Quebra_Antecipada` (rescisão antes do prazo contratual) ou `Proprietario_Evadiu` (baixa do imóvel pelo proprietário).
* **Algoritmos Disponíveis:**
  1. `Floresta Aleatória (Classificador)` (`RandomForestClassifier`)
  2. `Regressão Logística` (`LogisticRegression`)
  3. `Vizinhos Mais Próximos (KNN)` (`KNeighborsClassifier`)
  4. `Árvore de Decisão (Classificador)` (`DecisionTreeClassifier`)
* **Hiperparâmetros e Avaliação:** Permite calibrar em tempo real o parâmetro de regularização $C$ (Logística) ou número de vizinhos $K$ (KNN) de $1,0$ a $15,0$, além da profundidade máxima das árvores ($2$ a $15$), exibindo **Acurácia (%)**, **F1-Score (%)** e o mapa de calor da **Matriz de Confusão**.

### 7.3. Simulador Executivo de Novos Contratos & Retreinamento Dinâmico
* Permite selecionar simultaneamente o algoritmo regressor, o algoritmo classificador e o tipo de risco avaliado.
* Ao preencher os atributos de um novo contrato em negociação (Aluguel Líquido, Taxa de Condomínio, Área Total em $\text{m}^2$, Total de Quartos, Modalidade de Garantia, Política de Repasse, Bairro e Quantidade de Imóveis do Proprietário), o sistema monta o vetor padronizado e estima instantaneamente:
  1. A **Permanência Estimada em Meses** e o delta ($\pm$) em relação à média histórica da carteira.
  2. A **Probabilidade de Risco (%)** via `predict_proba`, classificando o cenário em *Risco Controlado* ou *Risco Elevado*.
* **Retreinamento Dinâmico:** Como o pipeline lê as bases atualizadas a cada ciclo de execução, qualquer novo upload em lote, cadastro avulso ou edição CRUD na aba de Ingestão recalibra imediatamente os modelos preditivos.

---

## 8. Pipeline Não Supervisionado (Clusterização, Anomalias e PCA)

Implementado em `src/controllers/ml_clusters_ctrl.py` e `src/views/clusters_view.py`, o módulo agrupa os contratos saneados com base em quatro dimensões contínuas padronizadas com `StandardScaler` ($\mu = 0, \sigma = 1$): Valor Líquido do Aluguel, Tempo de Permanência (`Meses_Locados`), Taxa de Condomínio e Porte do Proprietário (`Qtd_Imoveis_Proprietario`), após corte de valores extremos acima de $Q_3 + 3 \cdot \text{IQR}$.

* **Segmentação de Personas (K-Means):** Agrupa os contratos em $K$ clusters configuráveis ($K \in [2, 7]$) e avalia a coesão e separação geométrica pelo **Coeficiente de Silhueta Médio $S(i)$**, gerando uma tabela executiva com as médias financeiras, permanência e taxas de quebra/evasão de cada persona.
* **Isolamento de Anomalias por Densidade (DBSCAN):** Permite ajustar o raio de vizinhança `eps` ($0,5$ a $2,5$) e o mínimo de pontos `min_samples` ($2$ a $12$) para detectar automaticamente contratos atípicos (rotulados como `-1`) e listá-los em uma tabela de auditoria.
* **Redução de Dimensionalidade (PCA 2D):** Projeta as quatro variáveis padronizadas em dois componentes principais ortogonais (`Eixo PC1` e `Eixo PC2`), exibindo o percentual de variância explicada acumulada e plotando lado a lado os grupos do K-Means e as anomalias isoladas pelo DBSCAN.
* **Agrupamento Hierárquico Aglomerativo (Método de Ward):** Constrói a matriz de ligação (`scipy.cluster.hierarchy.linkage`) minimizando a variância intra-cluster (WCSS) e renderiza o **Dendrograma de Ward** truncado nas 25 ramificações superiores.

---

## 9. Requisitos, Configuração de Ambiente e Execução

### Pré-requisitos
* **Python:** Versão $3.10$ ou superior (compatível com a sintaxe estrutural `match/case`).
* **Gerenciador de Pacotes:** `pip` e módulo `venv`.

### Passo a Passo de Instalação

**1. Clonar o repositório e acessar o diretório raiz:**
```bash
git clone https://github.com/otavio-wenzel/projeto-baggio-analytics.git
cd projeto-baggio-analytics
```

**2. Criar e ativar o ambiente virtual (`.venv`):**
* No Windows (PowerShell / CMD):
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  ```
* No Linux ou macOS:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

**3. Instalar as dependências oficiais do projeto:**
```bash
pip install -r requirements.txt
```

**4. Configurar as variáveis de ambiente (`.env`):**
Copie o arquivo modelo `.env.example` para um novo arquivo `.env` na raiz do projeto:
* No Windows:
  ```bash
  copy .env.example .env
  ```
* No Linux ou macOS:
  ```bash
  cp .env.example .env
  ```

**5. Executar a aplicação web no Streamlit:**
```bash
streamlit run app.py
```
O navegador abrirá automaticamente no endereço local `http://localhost:8501`.

---

## 10. Guia de Operação e Manutenção da Base Local

1. **Alimentação Inicial (Aba "Ingestão de Dados"):**
   * Acesse o menu lateral e clique em **Ingestão de Dados**.
   * Arraste os arquivos de **Imóveis** e de **Contratos Rescindidos** (nos formatos `.xlsx`, `.xls`, `.csv`, `.json` ou `.xml`) para a área de upload e clique em **Validar Estrutura, Anonimizar (LGPD) e Salvar Lote**.
   * O sistema reconhecerá automaticamente cada tabela pela assinatura de colunas, aplicará o Hash SHA-256 nos documentos, descartará as colunas de PII e criará os arquivos `data_store/imoveis_anonimizados.json` e `data_store/contratos_anonimizados.json`.
2. **Cadastros Manuais e CRUD de Ciclo de Vida:**
   * Na mesma tela de Ingestão, expanda as seções **Inclusão Manual Avulsa** ou **Gestão de Ciclo de Vida do Imóvel (CRUD Discreto)** para cadastrar novos registros, registrar a baixa de um imóvel ou reativá-lo na carteira.
3. **Como Reiniciar / Zerar o Banco de Dados Local:**
   * Caso deseje limpar todos os dados persistidos para realizar uma demonstração do zero, basta excluir a pasta `data_store/` na raiz do projeto (ou apagar os arquivos `.json` dentro dela). Na inicialização seguinte, o módulo `src/models/repository.py` recriará o diretório automaticamente.

---

## 11. Matriz de Responsabilidades para Defesa de Código

A modularização em 14 arquivos sob o padrão MVC permite uma divisão clara de responsabilidades entre os integrantes da equipe durante a apresentação prática e defesa técnica:

| Frente Técnica | Módulos Responsáveis | Tópicos-Chave para Defesa de Código |
| :--- | :--- | :--- |
| **1. Arquitetura, Esquemas & Governança LGPD** | `app.py`<br>`src/models/schemas.py`<br>`src/models/lgpd_sanitizer.py` | Roteamento `match/case`, gerenciamento de estado (`st.session_state`), imutabilidade de tuplas, validação de payload por Conjuntos (`issubset`), hashing SHA-256 com *Salt* via `.env` e descarte de PII em memória RAM. |
| **2. Persistência Segura, Ingestão Multiformato & CRUD** | `src/models/repository.py`<br>`src/views/ingestao_view.py`<br>`src/views/ui_components.py` | Leitura flexível via `io.BytesIO`, sanitização contra *Path Traversal*, escrita atômica com arquivo temporário (`.tmp.json`), deduplicação incremental (*Upsert*), persistência de tema (`settings.json`) e formulários CRUD. |
| **3. Engenharia Relacional, Estatística IQR & Visualizações EDA** | `src/controllers/data_processor.py`<br>`src/controllers/eda_temporal_ctrl.py`<br>`src/controllers/eda_politicas_ctrl.py`<br>`src/views/dashboard_view.py`<br>`src/views/eda_view.py` | Cruzamento `pd.merge`, cálculo das variáveis-alvo, corte estatístico de *outliers* ($Q_3 + k \cdot \text{IQR}$), detecção matemática de picos sazonais ($\mu + k \cdot \sigma$), matriz de Pearson e compositor das 9 análises interativas. |
| **4. Machine Learning Supervisionado & Não Supervisionado** | `src/controllers/ml_supervisionado_ctrl.py`<br>`src/controllers/ml_clusters_ctrl.py`<br>`src/views/ml_view.py`<br>`src/views/clusters_view.py` | *One-Hot Encoding*, validação cruzada K-Fold contra *Baseline*, diagnóstico de *Overfitting/Underfitting*, classificadores de risco, Matriz de Confusão, Simulador Executivo, retreinamento dinâmico, K-Means (Silhueta), DBSCAN, PCA 2D e Ward. |

---

*Projeto desenvolvido como requisito de avaliação final da disciplina de Tópicos Especiais em Software.*