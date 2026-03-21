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

# Header de contexto
sys_prompt = (
            "system",
            "Você é um assistente virtual. Caso necessário, utilize ferramentas em suas respostas. \n"
            "Se necessário utilizar uma ferramente, use ela antes de responder. \n"
            "Se não souber a resposta, por favor, diga não saber e peça por mais informações ao usuário. \n"
            # "As ferramentas dir_* servem para visualização, locomoção e interação com arquivos e diretórios. \n"
            "Use ferramentas apenas quando necessário. \n"
            "Forneça apenas o mínimo de informação necessário, nada mais."
        )

ferramentas = [get_context, get_dt, dir_ls]
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
    resp = agente.invoke({"messages": prompt},
                       {"configurable": {"thread_id": "1"}})
    if sys_info:
        print("---\n")
        for m in resp['messages']:
            m.pretty_print()
        print("---\n")
    return resp['messages'][-1].content