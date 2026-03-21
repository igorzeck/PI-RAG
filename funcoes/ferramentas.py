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
from langchain_community.document_loaders import PDFMinerLoader
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

embeddings = OllamaEmbeddings(model="llama3")
vector_store = InMemoryVectorStore(embeddings)

# --- Funções ---
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
# Necessário especificar o funcionamento das ferramentas por meio do DOCSTRING
# Assim, o agente consegue melhor entender as funções e seus parâmetros
# Função de retrieval de documentos (PDF)
@tool(response_format="content_and_artifact")
def get_context(query: str):
    """Retrieval de informção para a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Função teste de coleta de dados .csv
@tool 
def abrir_arquivo(arquivo: str) -> str:
    """Abre o arquivo especificado e visualiza as primeiras 5 linhas."""
    # TODO: Wrapper para identificar os tipos de arquivo!
    pass

@tool
def get_dt(dataset: str = "", coluna: str = "") -> str:
    """Coleta conteúdo de um dataset"""
    return "12, 13, 14"

# - Funções dir -
@tool
def ler_arquivo(subcaminho: str) -> str:
    """
    Lê o conteúdo de um arquivo de texto.
    Forneça um caminho relativo como 'README.md' ou 'src/main.py'.
    """
    alvo = (HOMEDIR / subcaminho).resolve()

    if not str(alvo).startswith(str(HOMEDIR.resolve())):
        return "Erro: Acesso negado fora do diretório de trabalho."

    if not alvo.exists():
        return f"Erro: Arquivo '{subcaminho}' não encontrado."

    if not alvo.is_file():
        return f"Erro: '{subcaminho}' não é um arquivo."

    try:
        return alvo.read_text(encoding="utf-8")
    except Exception as e:
        return f"Erro ao ler o arquivo: {e}"
    
# -- Teste com busca navegável --
@tool
def buscar_docs(query: str, k: int = 5) -> str:
    """
    Busca documentos relevantes para a query.
    Retorna IDs que podem ser usados em `abrir_doc`.
    """
    docs = vector_store.similarity_search(query, k=k)

    results = []
    for i, doc in enumerate(docs):
        results.append(
            f"id:{i} | source:{doc.metadata} | preview:{doc.page_content[:200]}"
        )

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
    Lista arquivos de um diretório.
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

    entradas = [
        f"{'Diretório: ' if item.is_dir() else 'Documento: '} {item.name}"
        for item in sorted(alvo.iterdir())
    ]

    return f"Conteúdo de '{subcaminho}':\n" + "\n".join(entradas) if entradas else "Diretório vazio."
