# ---- Funções e ferramentas utilizadas ----
# Aqui também são definida as bibliotecas utilizadas
# Necessário ollama, mais sobre no README.md
# ---- Funções e ferramentas utilizadas ----

# TODO: Timeout tempo de resposta do bot
# region: Setup ---
from langchain.agents import AgentState
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# Conteúdo
# from langchain_community.document_loaders import WebBaseLoader  # Página web
# Loader para arquivo .pdf, .txt e .md
from langchain_community.document_loaders import PDFMinerLoader, TextLoader, UnstructuredMarkdownLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

import pandas as pd

from langchain.tools import tool
from langchain.agents.middleware import before_model

from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from typing import Any
# from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

# Outros
from pathlib import Path
from colorama import Fore
import pickle as pkl
import funcoes.churn_rate as churn_rate
# - Variáveis e Constantes -
MAX_MESSAGES = 100  # O ideal seria manter as mensagens do usuário!

#__file__ é o caminho para desse arquivo ("ferramentas.py"),
# o resolve pega o caminho absoluto e o parent, parent leva ao diretório base
BASE_DIR = Path(__file__).resolve().parent.parent
HOMEDIR = BASE_DIR / 'playground'

INDEXADOS = set()
DATASETS = {}  # Dict

# PATH para salvar/carreagr o modelo de Churn Rate
PATH_MODELO = Path("recursos/modelo_churn_rate.pkl")
PATH_DATASETS_CHRUN = Path("playground/datasets/Telco-Customer-Churn.csv")

# Necessário dar um pull nesse modelo de embedding!
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# De acordo com o LangChain, as métricas de similaridade podem ser:
# Cosine similarity
# Euclidean distance
# Dot product
# https://docs.langchain.com/oss/python/integrations/vectorstores
vector_store = InMemoryVectorStore(embeddings)

# Idealmente no script RAG
modo_db = False
# endregion: Setup ---

# region: Funções ---
def print_etapa_msg(type: str = "", msg: str = "", end="\n"):
    if type:
        print(Fore.LIGHTGREEN_EX + type + Fore.RESET,end=": " if msg else end)
    if msg:
        print(Fore.LIGHTBLACK_EX + msg + Fore.RESET, end=end)

def print_sys_msg(msg: str, type: str = "Info", end="\n", flush = False):
    cor = Fore.LIGHTYELLOW_EX
    if type == "Aviso":
        cor = Fore.LIGHTYELLOW_EX
    elif type == "Erro":
        cor = Fore.LIGHTRED_EX
    elif type == "OK":
        cor = Fore.LIGHTGREEN_EX
    elif type == "Info":
        # Padrão
        cor = Fore.LIGHTBLACK_EX

    if cor:
        print(f"{cor}{msg}{Fore.RESET}",end=end,flush=flush)
    else:
        print(msg,end="\n",flush=flush)


# region: Middleware --
@before_model
def trimming(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Mantém apenas as N últimas mensagens para manter na janela de contexto."""
    messages = state["messages"]

    if len(messages) <= MAX_MESSAGES:
        return None
    
    print_etapa_msg(f"Trimming - {len(messages)} > {MAX_MESSAGES}")
    first_msg = messages[0]
    recent_messages = messages[-MAX_MESSAGES:] if len(messages) % 2 == 0 else messages[-(MAX_MESSAGES + 1):]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }
# endregion: Middleware --

# region: Ferramentas --
# TODO: Função para "sair" do bot (reseta ele)
# - Base -
@tool
def finalizar_conversa() -> str:
    """
    Função para finalizar a conversa. Sempre chame essa função quando
    O usuário começar a falar de coisas não relacionadas aos arquivos.
    """
    print("O RAG finalizou este chat.")
    # Esdd
    exit(0)
# - Churn rate -
@tool
def calcular_churn_rate() -> str:
    """
    Calcula a taxa de Churn Rate e retorna seu valor.

    Essa função NÃO recebe parâmetros.

    Os valores são coletados pela própria função.
    """
    modelo = None
    # 1. Checa se já há algum modelo de Churn Rate treinado
    if not PATH_MODELO.is_file():
        modelo = churn_rate.treinar_modelo()

        # Salva o modelo
        with open(PATH_MODELO, 'wb') as arq:
            pkl.dump(modelo, arq)
    
    # 2. Coleta esse modelo
    if not modelo:
        with open(PATH_MODELO, 'rb') as arq:
            modelo = pkl.load(arq)
    
    # 3. Executa modelo para os dados disponíveis
    print_sys_msg(f"Modelo de Churn Rate encontrado com sucesso. Acurácia de teste: {modelo.melhor_acc * 100:.2f}%", type="OK")
    
    df_proc = pd.read_csv(PATH_DATASETS_CHRUN).drop(columns=churn_rate.TARGET_COLUMN)

    # Pré-processamento do dataset
    # TODO: Função a parte
    df_proc['TotalCharges'] = pd.to_numeric(df_proc['TotalCharges'], errors='coerce')
    df_proc['TotalCharges'].fillna(df_proc['TotalCharges'].median(), inplace=True)

    df_proc.drop(columns=['customerID'], inplace=True, errors='ignore')

    le = churn_rate.LabelEncoder()
    for col in df_proc.columns:
        if df_proc[col].dtype == 'object' or str(df_proc[col].dtype) == 'bool':
            df_proc[col] = le.fit_transform(df_proc[col].astype(str))

    X = df_proc.values.astype(float)

    resultado = modelo.predict(X_new=X)
    prop_s = sum(resultado) / len(resultado)
    prop_n = 1 - prop_s
    # Previsão dos dados
    return f"Previsão de Chrun (dados: '{PATH_DATASETS_CHRUN.name}'):\nNão: {prop_n*100:.2f} ({len(resultado) - sum(resultado)})\nSim: {prop_s*100:.2f} ({sum(resultado)})\n"


# - Com busca navegável (MODO EXPERIMENTAL) -
# As funções abaixo utilizam busca navegável
# e se ecnontram em fase experimental

@tool
def buscar_docs(query: str, k: int = 5) -> str:
    """
    Busca documentos relevantes para a query.
    Retorna PIDs que podem ser usados em `abrir_doc`.
    Utilize essa ferramenta para arquivos de texto JÁ INDEXADOS. Em caso de dataset
    utilize 'listar_diretorio' para encontrar o nome do dataset indexado e então utilize 'dataset_query' ou 'dataset_info'.
    """
    # TODO: Fazer retornar apenas se o vector_store tiver algo!
    docs = vector_store.similarity_search(query, k=k)

    results = []
    for i, doc in enumerate(docs):
        results.append(
            f"pid:{i} | source:{doc.metadata} | preview:{doc.page_content[:200]}"
        )

    if not results:
        results.append("Nennhum documento encontrado. Verifique se os documentos relevantes foram indexados com as funções do tipo indexar_*")

    return "\n".join(results)

@tool
def abrir_doc(doc_pid: int) -> str:
    """
    Abre um documento retornado pela ferramenta buscar_docs.
    """
    docs = vector_store.similarity_search("", k=10)

    if doc_pid >= len(docs):
        return "Documento não encontrado. Tem certeza que ele foi indexado com 'indexar_documentos' e listado com 'buscar_docs'?"

    return docs[doc_pid].page_content

@tool
def listar_diretorio(subcaminho: str = ".") -> str:
    """
    Lista arquivos e subdiretórios de um subcaminho.
    Use '.' para o diretório atual.
    Senão, forneça o nome completo do subcaminho.
    Retorna o nome dos arquivos e se eles foram ou não indexados.
    Exemplo:
        0. Arquivo1.csv (Indexado: Não)
        1. Arquivo2.txt (Indexado: Não)
        2. Arquivo3.txt (Indexado: Sim)

    Use esta ferramenta antes de `indexar_documento` ou `indexar_dataset
    para descobrir quais arquivos e diretórios existem.
    """

    alvo = (HOMEDIR / subcaminho).resolve()

    if not str(alvo).startswith(str(HOMEDIR.resolve())):
        return "Erro: Acesso negado fora do diretório de trabalho."

    if not alvo.exists():
        return f"Erro: O caminho '{subcaminho}' não existe."

    if not alvo.is_dir():
        return f"Erro: '{subcaminho}' não é um diretório."

    entradas = []
    for i, item in enumerate(sorted(alvo.iterdir())):
        if item.is_dir():
            entradas.append(f"{i}. {item.name}/")
        else:
            entradas.append(f"{i}. {item.name} (Indexado: {"Sim" if str(item.resolve()) in INDEXADOS else "Não"})")
    return f"Conteúdo de '{subcaminho}':\n" + "\n".join(entradas) if entradas else "Diretório vazio."

@tool
def indexar_documento(subcaminho: str) -> str:
    """
    Indexa um documento no vector store.
    OU indexa um dataset em um dicionário de datasets. 
    
    Formatos suportados: (PDF, TXT, Markdown, CSV)

    Argumentos:
    - subcaminho (str): O NOME exato do arquivo que você quer indexar (exemplo: 'p.txt', 'example.pdf').
    NUNCA use '.' ou chute nomes de arquivos aqui. Use nomes exatos descobertos com 'listar_diretorio'.
    """

    # TODO: Tirar RESOLVE do Paths
    alvo = (HOMEDIR / subcaminho).resolve()

    if not str(alvo).startswith(str(HOMEDIR.resolve())):
        return "Erro: acesso negado fora do diretório de trabalho."

    if not alvo.exists():
        return f"Erro: arquivo '{subcaminho}' não encontrado."

    if not alvo.is_file():
        return f"Erro: '{subcaminho}' não é um arquivo."

    # evita indexar duas vezes
    if str(alvo) in INDEXADOS:
        return f"O arquivo '{subcaminho}' já foi indexado."

    extencao = alvo.suffix.lower()

    try:

        if extencao == ".pdf":
            loader = PDFMinerLoader(str(alvo))

        elif extencao == ".txt":
            loader = TextLoader(str(alvo), encoding="utf-8")

        elif extencao in [".md", ".markdown"]:
            loader = UnstructuredMarkdownLoader(str(alvo))
        
        # - Datasets -
        elif extencao == ".csv":
            df = pd.read_csv(alvo)
            INDEXADOS.add(str(alvo))
            DATASETS[subcaminho] = df
            return (
                        f"Dataset '{subcaminho}' indexado.\n"
                        f"Linhas: {len(df)}\n"
                        f"Colunas: {list(df.columns)}"
                    )

        else:
            return f"Formato '{extencao}' não suportado. Talvez seja possível com indexar_dataset?"

        docs = loader.load()

        # Metadata provavelmente desnecesário se for um .pdf
        metadata = {
            "source": str(alvo),
            "filename": alvo.name
        }

        for doc in docs:
            doc.metadata.update(metadata)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        vector_store.add_documents(chunks)

        INDEXADOS.add(str(alvo))

        return f"Documento '{subcaminho}' indexado ({len(chunks)} trechos)."

    except Exception as e:
        return f"Erro ao indexar documento: {e}"

# -- Arquivos CSV --
@tool
def dataset_info(dataset: str) -> str:
    """
    Mostra informações sobre um dataset indexado:
    colunas, tipos e primeiras linhas.
    dataset: str - subcaminho do dataset indexado.
    """

    if dataset not in DATASETS:
        return "Dataset não indexado."

    df = DATASETS[dataset]

    preview = df.head(5).to_string()

    return (
        f"Dataset: {dataset}\n"
        f"Linhas: {len(df)}\n"
        f"Colunas: {list(df.columns)}\n\n"
        f"Tipos:\n{df.dtypes}\n\n"
        f"Amostra:\n{preview}"
    )


@tool
def dataset_query(dataset: str, operacao: str, coluna: str = "") -> str:
    """
    Executa operações simples em um dataset.
    dataset: str - subcaminho do dataset indexado.
    operacao: str - Operação a ser executada.
    coluna: str - Coluna a ser utilizada na operação.

    Operações suportadas:
    - mean
    - sum
    - max
    - min
    - unique
    - describe
    """

    if dataset not in DATASETS:
        return "Dataset não indexado."

    df = DATASETS[dataset]

    try:

        if operacao == "mean":
            return str(df[coluna].mean())

        elif operacao == "sum":
            return str(df[coluna].sum())

        elif operacao == "max":
            return str(df[coluna].max())

        elif operacao == "min":
            return str(df[coluna].min())

        elif operacao == "unique":
            return str(df[coluna].unique())

        elif operacao == "describe":
            return str(df.describe())

        else:
            return "Operação não suportada."

    except Exception as e:
        return f"Erro ao executar operação: {e}"
# endregion: Ferramentas --
# endregion: Funcções ---=