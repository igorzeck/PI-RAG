# Funções e ferramentas utilizadas
# Aqui também são definida as bibliotecas utilizadas
# Necessário ollama, mas sobre no README.md
# Pelo chatbot
# -- Setup --
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

# - Variáveis e Constantes -
MAX_MESSAGES = 3

embeddings = OllamaEmbeddings(model="llama3")
vector_store = InMemoryVectorStore(embeddings)

# -- Funções --
# TODO: Middleware ou Tools para poder "ver" o diretório?

@before_model
def trimming(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Mantém apenas as N últimas mensagens para manter na janela de contexto."""
    messages = state["messages"]

    if len(messages) <= MAX_MESSAGES:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-MAX_MESSAGES:] if len(messages) % 2 == 0 else messages[-(MAX_MESSAGES + 1):]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

@before_model
def get_dir(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """visualizador do diretório atual."""
    messages = state["messages"]

    if len(messages) <= MAX_MESSAGES:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-MAX_MESSAGES:] if len(messages) % 2 == 0 else messages[-(MAX_MESSAGES + 1):]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

# - Ferramentas -
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
def get_dt(dataset: str = "", coluna: str = "") -> str:
    """Coleta conteúdo de um dataset"""
    return "12, 13, 14"