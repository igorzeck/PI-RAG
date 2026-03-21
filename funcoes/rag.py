# --- RAG ---

# -- Setup --
from funcoes.ferramentas import *  # Feito por brevidade

# - Imports do RAG -
from langchain.agents import create_agent
from typing import TypedDict, Sequence
from langchain_core.messages import BaseMessage

# Memory saver para memória de curto prazo
from langgraph.checkpoint.memory import MemorySaver

# - Criação do Agente e seus componentes -
llm = ChatOllama(model="qwen2.5:0.5b")

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]

sys_prompt = """
Você é um assistente virtual.
Caso necessário, utilize ferramentas.

Você pode usar ferramentas múltiplas vezes
até obter todas as informações necessárias.
"""

ferramentas = [listar_diretorio, buscar_docs, abrir_doc]
middleware = [trimming]

agente = create_agent(
    llm,
    tools=ferramentas,
    system_prompt=sys_prompt,
    checkpointer=MemorySaver(),
    middleware=middleware
)

# -- Funções --
# - Funções de auxílio -
def e_valido(prompt: str) -> bool:
    """Retorna se o prompt é valido para ser passado para o agente."""
    return (prompt != "")

# - Funções de chamada -
# Aqui se encontra as funções para acessar o RAG em si
def conv(prompt: str = "", sys_info = False) -> str:
    resp = agente.invoke({"messages": [{"role": "user", "content": prompt}]},
                       {"configurable": {"thread_id": "1", "recursion_limit": 20}})
    if sys_info:
        print("---\n")
        for m in resp['messages']:
            m.pretty_print()
        print("---\n")
    return resp['messages'][-1].content