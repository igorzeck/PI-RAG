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
HOME = Path('playground')
CURDIR = Path('playground')  # Idealmente deveria estar em rag.py

embeddings = OllamaEmbeddings(model="llama3")
vector_store = InMemoryVectorStore(embeddings)

# --- Funções ---
# - Auxiliary -
def _iter_dir(diretorio: str = CURDIR) -> str:
    head = f"Conteúdo do diretório '{diretorio.name if diretorio.name else '.'}':\n"
    itens = ""
    for item in diretorio.iterdir():
        # TODO: Consertar ifs abaixo
        if item.is_dir():
            itens += "Diretório: "
        elif item.is_file():
            itens += 'Arquivo: '
        
        itens += item.name + "\n"
    if not itens:
        itens = 'Vazio'
    return head + itens


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

# Por agora, a visualização do que está no diretório atual é por meio de um middleware
@before_model
def dir_ls(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Exibe o conteúdo do diretório atual."""
    pass


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
# O ideal é que o agente iterasse essas funções até conseguir o que deseja!
# @tool
# def dir_ls() -> str:
#     """Lista tudo no diretório atual."""
#     return _iter_dir()

# TBA
# @tool
# def dir_cd(diretorio_alvo: str) -> str:
#     """Entra no diretório especifico."""
#     global CURDIR

#     # Limpa string
#     diretorio_alvo = re.sub(r'[^a-zA-Z0-9]', '', diretorio_alvo)

#     itens = f"Entrando em {diretorio_alvo}\n"
#     novo_dir = Path(HOME / diretorio_alvo)
#     if novo_dir.exists():
#         itens += _iter_dir(novo_dir)
#         CURDIR = novo_dir
#     else:
#         itens += "Diretório inexistente."
#     return itens