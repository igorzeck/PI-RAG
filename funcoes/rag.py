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
            "Use ferramentas apenas quando necessário. \n"
            "Se receber uma mensagem em branco ou não entender a pergunta, fale bom dia."
        )

ferramentas = [get_context, get_dt]
middleware = [trimming, get_dir]

agente = create_agent(
    llm,
    tools=ferramentas,
    system_prompt=sys_prompt,
    checkpointer=MemorySaver(),
    middleware=middleware
)

# -- Funções de chamada --
# Aqui se encontra as funções para acessar o RAG em si
def conv(prompt: str = "") -> str:
    resp = agente.invoke({"messages": prompt},
                        {"configurable": {"thread_id": "1"}})
    print(resp)
    return resp["messages"][-1].content