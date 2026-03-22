# Funções e ferramentas utilizadas
# Aqui também são definida as bibliotecas utilizadas
# Necessário ollama, mas sobre no README.md
# Pelo chatbot
# --- Setup ---
# from langchain.chat_models import init_chat_model
from langchain.agents import AgentState

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore

# Conteúdo
# from langchain_community.document_loaders import WebBaseLoader  # Página web
# Loader para arquivo .txt e .md
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

from pathlib import Path
import re

# - Variáveis e Constantes -
MAX_MESSAGES = 5
HOMEDIR = Path('playground')  # Idealmente deveria estar em rag.py
INDEXADOS = set()
DATASETS = {}  # Dict

# Necessário dar um pull nesse modelo de embedding!
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = InMemoryVectorStore(embeddings)


# --- Funções ---
# -- Auxiliares --


# -- Middleware --
@before_model
def trimming(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Mantém apenas as N últimas mensagens para manter na janela de contexto."""
    messages = state["messages"]

    if len(messages) <= MAX_MESSAGES:
        return None
    
    first_msg = messages[0]
    recent_messages = messages[-MAX_MESSAGES:] if len(messages) % 2 == 0 else messages[-(MAX_MESSAGES + 1):]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

# -- Ferramentas --
# -- Teste com busca navegável --
@tool
def buscar_docs(query: str, k: int = 5) -> str:
    """
    Busca documentos relevantes para a query.
    Retorna IDs que podem ser usados em `abrir_doc`.
    """
    # TODO: Fazer retornar apenas se o vector_store tiver algo!
    docs = vector_store.similarity_search(query, k=k)

    results = []
    for i, doc in enumerate(docs):
        results.append(
            f"id:{i} | source:{doc.metadata} | preview:{doc.page_content[:200]}"
        )

    if not results:
        results.append("Nennhum documento encontrado. Verifique se os documentos relevantes foram indexados com as funções do tipo indexar_*")

    return "\n".join(results)

@tool
def abrir_doc(doc_id: int) -> str:
    """
    Abre um documento retornado pela ferramenta buscar_docs.
    """
    docs = vector_store.similarity_search("", k=10)

    if doc_id >= len(docs):
        return "Documento não encontrado."

    return docs[doc_id].page_content

@tool
def listar_diretorio(subcaminho: str = ".") -> str:
    """
    Lista arquivos e subdiretórios de um subcaminho.
    Use '.' para o diretório atual.

    Use esta ferramenta antes de `ler_arquivo`
    para descobrir quais arquivos existem.
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

        elif item.suffix.lower() == ".pdf":
            entradas.append(f"{i}. {item.name} (use indexar_pdf)")

        else:
            entradas.append(f"{i}. {item.name}")

    return f"Conteúdo de '{subcaminho}':\n" + "\n".join(entradas) if entradas else "Diretório vazio."

@tool
def indexar_documento(subcaminho: str) -> str:
    """
    Indexa um documento no vector store.

    Formatos suportados atualmente:
    - PDF
    - TXT
    - Markdown (.md)

    Use esta ferramenta antes de realizar busca semântica em documentos.
    Caso o arquivo esteja no diretório raiz '.', passe apenas o nome dele.
    """

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

    ext = alvo.suffix.lower()

    try:

        if ext == ".pdf":
            loader = PDFMinerLoader(str(alvo))

        elif ext == ".txt":
            loader = TextLoader(str(alvo), encoding="utf-8")

        elif ext in [".md", ".markdown"]:
            loader = UnstructuredMarkdownLoader(str(alvo))

        else:
            return f"Formato '{ext}' não suportado. Talvez seja possível com indexar_dataset?"

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
import pandas as pd

@tool
def indexar_dataset(subcaminho: str) -> str:
    """
    Indexa um dataset CSV para análise.

    Após indexar, o dataset pode ser consultado usando:
    - dataset_info
    - dataset_query
    """

    alvo = (HOMEDIR / subcaminho).resolve()

    if not str(alvo).startswith(str(HOMEDIR.resolve())):
        return "Erro: acesso negado."

    if not alvo.exists():
        return f"Erro: arquivo '{subcaminho}' não encontrado."

    if alvo.suffix.lower() != ".csv":
        return "Erro: apenas arquivos CSV são suportados."

    try:
        df = pd.read_csv(alvo)

        DATASETS[subcaminho] = df

        return (
            f"Dataset '{subcaminho}' indexado.\n"
            f"Linhas: {len(df)}\n"
            f"Colunas: {list(df.columns)}"
        )

    except Exception as e:
        return f"Erro ao carregar dataset: {e}"
    
@tool
def dataset_info(dataset: str) -> str:
    """
    Mostra informações sobre um dataset indexado:
    colunas, tipos e primeiras linhas.
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